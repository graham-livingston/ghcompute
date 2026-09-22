using System;
using System.Collections.Generic;

using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;

namespace GHCompute
{
  public class GpuKnnComponent : GH_Component
  {
    public GpuKnnComponent()
      : base("GPU k-NN", "GpuKnn",
        "k-nearest-neighbor search on a point cloud, GPU-accelerated via PyTorch.",
        "GHCompute", "PointCloud")
    {
    }

    protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
    {
      pManager.AddPointParameter("Points", "P", "Input point cloud", GH_ParamAccess.list);
      pManager.AddIntegerParameter("k", "k", "Number of neighbors", GH_ParamAccess.item, 8);
    }

    protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
    {
      pManager.AddIntegerParameter("Indices", "I", "Neighbor indices, k per input point", GH_ParamAccess.tree);
      pManager.AddNumberParameter("Distances", "D", "Neighbor distances, k per input point", GH_ParamAccess.tree);
      pManager.AddTextParameter("Device", "Dev", "Backend device used (cpu, mps, cuda)", GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
      var points = new List<Point3d>();
      int k = 8;
      if (!DA.GetDataList(0, points)) return;
      if (!DA.GetData(1, ref k)) return;

      if (points.Count == 0)
      {
        AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Empty point list.");
        return;
      }
      if (k < 1 || k >= points.Count)
      {
        AddRuntimeMessage(GH_RuntimeMessageLevel.Error, $"k must be in [1, {points.Count - 1}], got {k}.");
        return;
      }

      try
      {
        var result = PythonBridge.Knn(points, k);
        int n = result.Indices.GetLength(0);
        int kOut = result.Indices.GetLength(1);

        var indicesTree = new GH_Structure<GH_Integer>();
        var distancesTree = new GH_Structure<GH_Number>();
        for (int i = 0; i < n; i++)
        {
          var path = new GH_Path(i);
          for (int j = 0; j < kOut; j++)
          {
            indicesTree.Append(new GH_Integer(result.Indices[i, j]), path);
            distancesTree.Append(new GH_Number(result.Distances[i, j]), path);
          }
        }
        DA.SetDataTree(0, indicesTree);
        DA.SetDataTree(1, distancesTree);
        DA.SetData(2, result.Device);

        Message = $"{result.Device} · {result.ElapsedMs:F0}ms";
      }
      catch (Exception ex)
      {
        AddRuntimeMessage(GH_RuntimeMessageLevel.Error, ex.Message);
      }
    }

    protected override System.Drawing.Bitmap Icon => null;
    public override Guid ComponentGuid => new Guid("27e5b35c-5b65-4330-92f7-0f6e77ea99c8");
  }
}
