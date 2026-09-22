using System;
using System.Drawing;
using Grasshopper;
using Grasshopper.Kernel;

namespace GHCompute
{
  public class GHComputeInfo : GH_AssemblyInfo
  {
    public override string Name => "GHCompute Info";

    //Return a 24x24 pixel bitmap to represent this GHA library.
    public override Bitmap Icon => null;

    //Return a short string describing the purpose of this GHA library.
    public override string Description => "";

    public override Guid Id => new Guid("71e34051-7096-4bc4-8973-aedec65fccde");

    //Return a string identifying you or your company.
    public override string AuthorName => "Graham Livingston";

    //Return a string representing your preferred contact details.
    public override string AuthorContact => "graham.m.livingston@gmail.com";

    //Return a string representing the version.  This returns the same version as the assembly.
    public override string AssemblyVersion => GetType().Assembly.GetName().Version.ToString();
  }
}