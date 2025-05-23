#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_PhysBCFunct.H>

using namespace amrex;

struct QGDBCFill 
{
    AMREX_GPU_DEVICE
    void operator() (const IntVect& /*iv*/, Array4<Real> const& /*dest*/,
                     int /*dcomp*/, int /*numcomp*/,
                     GeometryData const& /*geom*/, Real /*time*/,
                     const BCRec* /*bcr*/, int /*bcomp*/,
                     int /*orig_comp*/) const
        {
            // no physical boundaries to fill because it is all periodic
        }

};

void bcfill (Box const& bx, FArrayBox& data,
             int dcomp, int numcomp,
             Geometry const& geom, Real time,
             const Vector<BCRec>& bcr, int bcomp,int scomp)
{
    GpuBndryFuncFab<QGDBCFill> gpu_bndry_func(QGDBCFill{});
    gpu_bndry_func(bx,data,dcomp,numcomp,geom,time,bcr,bcomp,scomp);
}