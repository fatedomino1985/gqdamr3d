#include "AmrQGD.H"
#include <cmath>

using namespace amrex;

void AmrQGD::initData ()
{
	//std::cout << " QGD_init.cpp initData() - 1\n";
    const auto problo = Geom().ProbLoArray();
    const auto dx = Geom().CellSizeArray();
    MultiFab& S_new = get_new_data(State_Type);
    auto const& snew = S_new.arrays();

    amrex::ParallelFor(S_new,
    [=] AMREX_GPU_DEVICE (int bi, int i, int j, int k) noexcept
    {
		///Not to use it in your program?
        //Set Sc number
        /*snew[bi](i,j,k,0) = rhou;      //rho
        snew[bi](i,j,k,1) = Uu;        //Ux
        snew[bi](i,j,k,2) = Vu;        //Uy
        snew[bi](i,j,k,3) = pu;        //P
        snew[bi](i,j,k,4) = ScQgd;     //Sc
        snew[bi](i,j,k,5) = curl;      //vorticity
        snew[bi](i,j,k,6) = magGradRho;*/
        
		snew[bi](i,j,k,0) = rho;      //rho
        snew[bi](i,j,k,1) = rho_a;    //rho_a
        snew[bi](i,j,k,2) = rho_b;    //rho_b
        snew[bi](i,j,k,3) = ux;        //Ux
        snew[bi](i,j,k,4) = uy;        //Uy,Vu
        snew[bi](i,j,k,5) = uz;        //Uz,Wu
        snew[bi](i,j,k,6) = pu;        //P
        snew[bi](i,j,k,7) = E;         //E
        snew[bi](i,j,k,8) = E_in;      //E_in
        snew[bi](i,j,k,9) = E_in0;     //E_in0
        snew[bi](i,j,k,10) = T;        //T
        snew[bi](i,j,k,11) = Cs;       //Cs
        snew[bi](i,j,k,12) = Va;       //Va
        snew[bi](i,j,k,13) = Vb;       //Vb
		
    });
    FillPatcherFill(S_new, 0, ncomp, nghost, 0, State_Type, 0); 
}

