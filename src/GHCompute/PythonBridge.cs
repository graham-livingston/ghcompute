using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Text.Json;

using Rhino.Geometry;

namespace GHCompute
{
  internal static class PythonBridge
  {
    private const string EnvVar = "GHCOMPUTE_PYTHON";
    private const byte OpKnn = 0x01;
    private const byte OpVoxel = 0x02;
    private const byte StatusOk = 0x00;
    private const byte StatusUserError = 0x01;
    private const byte StatusInternalError = 0xFF;

    private static readonly string[] DefaultPaths =
    {
      "/usr/local/bin/python3",
      "/opt/homebrew/bin/python3",
      "/usr/bin/python3"
    };

    public sealed record KnnResult(int[,] Indices, float[,] Distances, string Device, double ElapsedMs);

    public sealed record VoxelResult(Point3d[] Points, string Device, double ElapsedMs);

    public static string ResolvePython()
    {
      var fromEnv = Environment.GetEnvironmentVariable(EnvVar);
      if (!string.IsNullOrEmpty(fromEnv) && File.Exists(fromEnv))
        return fromEnv;
      foreach (var p in DefaultPaths)
        if (File.Exists(p)) return p;
      throw new InvalidOperationException(
        $"Could not find python3. Set {EnvVar} to override.");
    }

    public static KnnResult Knn(IReadOnlyList<Point3d> points, int k)
    {
      var args = JsonSerializer.Serialize(new Dictionary<string, object> { ["k"] = k });
      var (meta, payload) = Invoke(OpKnn, args, EncodePoints(points));

      int n = meta.RootElement.GetProperty("shape")[0].GetInt32();
      int kOut = meta.RootElement.GetProperty("shape")[1].GetInt32();
      int idxBytes = n * kOut * sizeof(int);
      int distBytes = n * kOut * sizeof(float);
      if (payload.Length != idxBytes + distBytes)
        throw new InvalidOperationException(
          $"knn payload size mismatch: expected {idxBytes + distBytes}, got {payload.Length}");

      var indices = new int[n, kOut];
      Buffer.BlockCopy(payload, 0, indices, 0, idxBytes);
      var distances = new float[n, kOut];
      Buffer.BlockCopy(payload, idxBytes, distances, 0, distBytes);

      return new KnnResult(
        indices,
        distances,
        meta.RootElement.GetProperty("device").GetString() ?? "?",
        meta.RootElement.GetProperty("elapsed_ms").GetDouble()
      );
    }

    public static VoxelResult VoxelDownsample(IReadOnlyList<Point3d> points, double voxelSize)
    {
      var args = JsonSerializer.Serialize(new Dictionary<string, object> { ["size"] = voxelSize });
      var (meta, payload) = Invoke(OpVoxel, args, EncodePoints(points));

      int m = meta.RootElement.GetProperty("shape")[0].GetInt32();
      if (payload.Length != m * 3 * sizeof(float))
        throw new InvalidOperationException(
          $"voxel payload size mismatch: expected {m * 3 * sizeof(float)}, got {payload.Length}");

      var floats = new float[m * 3];
      Buffer.BlockCopy(payload, 0, floats, 0, payload.Length);
      var pts = new Point3d[m];
      for (int i = 0; i < m; i++)
        pts[i] = new Point3d(floats[i * 3], floats[i * 3 + 1], floats[i * 3 + 2]);

      return new VoxelResult(
        pts,
        meta.RootElement.GetProperty("device").GetString() ?? "?",
        meta.RootElement.GetProperty("elapsed_ms").GetDouble()
      );
    }

    private static byte[] EncodePoints(IReadOnlyList<Point3d> points)
    {
      var buf = new byte[points.Count * 3 * sizeof(float)];
      var span = buf.AsSpan();
      for (int i = 0; i < points.Count; i++)
      {
        var p = points[i];
        BinaryPrimitives.WriteSingleLittleEndian(span.Slice(i * 12 + 0, 4), (float)p.X);
        BinaryPrimitives.WriteSingleLittleEndian(span.Slice(i * 12 + 4, 4), (float)p.Y);
        BinaryPrimitives.WriteSingleLittleEndian(span.Slice(i * 12 + 8, 4), (float)p.Z);
      }
      return buf;
    }

    private static (JsonDocument Meta, byte[] Payload) Invoke(byte opcode, string argsJson, byte[] payload)
    {
      var python = ResolvePython();
      var psi = new ProcessStartInfo
      {
        FileName = python,
        ArgumentList = { "-m", "ghcompute.service" },
        RedirectStandardInput = true,
        RedirectStandardOutput = true,
        RedirectStandardError = true,
        UseShellExecute = false,
        CreateNoWindow = true
      };

      using var proc = Process.Start(psi)
        ?? throw new InvalidOperationException("Process.Start returned null.");

      WriteRequestFrame(proc.StandardInput.BaseStream, opcode, argsJson, payload);
      proc.StandardInput.BaseStream.Flush();
      proc.StandardInput.Close();

      var (status, metaJson, responsePayload) = ReadResponseFrame(proc.StandardOutput.BaseStream);
      var stderr = proc.StandardError.ReadToEnd();
      if (!proc.WaitForExit(60_000))
      {
        try { proc.Kill(); } catch { }
        throw new TimeoutException("Python service exceeded 60s.");
      }

      var meta = JsonDocument.Parse(metaJson);
      if (status == StatusOk) return (meta, responsePayload);

      var message = meta.RootElement.TryGetProperty("message", out var m) ? m.GetString() : "(no message)";
      var prefix = status == StatusUserError ? "user error" : "internal error";
      var stderrSuffix = string.IsNullOrEmpty(stderr) ? "" : $"\nstderr: {stderr.Trim()}";
      throw new InvalidOperationException($"Python service returned {prefix}: {message}{stderrSuffix}");
    }

    private static void WriteRequestFrame(Stream stream, byte opcode, string argsJson, byte[] payload)
    {
      var argsBytes = Encoding.UTF8.GetBytes(argsJson);
      Span<byte> u32 = stackalloc byte[4];

      stream.WriteByte(opcode);
      BinaryPrimitives.WriteUInt32BigEndian(u32, (uint)argsBytes.Length);
      stream.Write(u32);
      stream.Write(argsBytes, 0, argsBytes.Length);
      BinaryPrimitives.WriteUInt32BigEndian(u32, (uint)payload.Length);
      stream.Write(u32);
      stream.Write(payload, 0, payload.Length);
    }

    private static (byte Status, string MetaJson, byte[] Payload) ReadResponseFrame(Stream stream)
    {
      Span<byte> single = stackalloc byte[1];
      ReadExact(stream, single);
      byte status = single[0];

      Span<byte> u32 = stackalloc byte[4];
      ReadExact(stream, u32);
      int metaLen = (int)BinaryPrimitives.ReadUInt32BigEndian(u32);
      var metaBytes = new byte[metaLen];
      ReadExact(stream, metaBytes);

      ReadExact(stream, u32);
      int payloadLen = (int)BinaryPrimitives.ReadUInt32BigEndian(u32);
      var payload = new byte[payloadLen];
      ReadExact(stream, payload);

      return (status, Encoding.UTF8.GetString(metaBytes), payload);
    }

    private static void ReadExact(Stream stream, Span<byte> dst)
    {
      int read = 0;
      while (read < dst.Length)
      {
        int n = stream.Read(dst.Slice(read));
        if (n <= 0) throw new EndOfStreamException(
          $"expected {dst.Length} bytes, got {read} before EOF");
        read += n;
      }
    }
  }
}
