using System;
using System.Collections.Generic;

using Grasshopper.Kernel;
using Rhino.Geometry;

namespace GHCompute
{
  public class GpuVoxelDownsampleComponent : GH_Component
  {
    public GpuVoxelDownsampleComponent()
      : base("GPU Voxel Downsample", "GpuVoxel",
        "Voxel-grid downsampling of a point cloud, GPU-accelerated via PyTorch.",
        "GHCompute", "PointCloud")
    {
    }

    protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
    {
      pManager.AddPointParameter("Points", "P", "Input point cloud", GH_ParamAccess.list);
      pManager.AddNumberParameter("Voxel Size", "S", "Voxel edge length, model units", GH_ParamAccess.item, 1.0);
    }

    protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
    {
      pManager.AddPointParameter("Points", "P", "Downsampled centroids", GH_ParamAccess.list);
      pManager.AddTextParameter("Device", "Dev", "Backend device used (cpu, mps, cuda)", GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
      var points = new List<Point3d>();
      double voxelSize = 1.0;
      if (!DA.GetDataList(0, points)) return;
      if (!DA.GetData(1, ref voxelSize)) return;

      if (points.Count == 0)
      {
        AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Empty point list.");
        return;
      }
      if (voxelSize <= 0)
      {
        AddRuntimeMessage(GH_RuntimeMessageLevel.Error, $"Voxel size must be > 0, got {voxelSize}.");
        return;
      }

      try
      {
        var result = PythonBridge.VoxelDownsample(points, voxelSize);
        DA.SetDataList(0, result.Points);
        DA.SetData(1, result.Device);
        Message = $"{result.Device} · {result.Points.Length} pts · {result.ElapsedMs:F0}ms";
      }
      catch (Exception ex)
      {
        AddRuntimeMessage(GH_RuntimeMessageLevel.Error, ex.Message);
      }
    }

    protected override System.Drawing.Bitmap Icon => null;
    public override Guid ComponentGuid => new Guid("cf2d16ea-5677-4554-a32d-5469f255e1fe");
  }
}
