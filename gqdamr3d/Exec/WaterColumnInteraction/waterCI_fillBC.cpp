#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_PhysBCFunct.H>

//
#include "AmrQGD.H" 

using namespace amrex;

struct QGDBCFill 
{
        AMREX_GPU_DEVICE
        void operator() (const IntVect& iv, Array4<Real> const& dest,
                         const int /*dcomp*/, const int /*numcomp*/,
                         GeometryData const& geom, const Real /*time*/,
                         const BCRec* /*bcr*/, const int /*bcomp*/,
                         const int /*orig_comp*/) const
        {
            const int ilo = geom.Domain().smallEnd(0);
            const int ihi = geom.Domain().bigEnd(0);
            const int jlo = geom.Domain().smallEnd(1);
            const int jhi = geom.Domain().bigEnd(1);
            const int klo = geom.Domain().smallEnd(2);
            const int khi = geom.Domain().bigEnd(2);

			
            const auto problo = geom.ProbLo();//data();
            const auto [i,j,k] = iv.dim3();

             // const auto problo = data1.ProbLo();
            const auto dx = geom.CellSize();

			int ir = 0, ira = 1, irb = 2, 
			iux = 3, iuy = 4, iuz = 5,
			ip = 6, iE = 7, iE_in = 8, iE_in0 = 9,
			iT = 10, 
			iCs = 11,
			iVa = 12, iVb = 13;
			//{
				//%%3.5 Shock/Water-Column interaction [3]Китамура
				double Runiv = AmrQGD::Runiv;
				double gma = AmrQGD::gamma_a, gmb = AmrQGD::gamma_b;
			
				double Ra = AmrQGD::RGas_a;
				double Rb = AmrQGD::RGas_b;
				double cva = Ra / (gma - 1); // 717.5 Air
				double cvb = Rb / (gmb - 1); // 1495 Water
				double pL = AmrQGD::pL,
				pR = AmrQGD::pR;
				double pa_inf = AmrQGD::painf,
				pb_inf = AmrQGD::pbinf;
				double T1 = AmrQGD::T1,
				T2 = AmrQGD::T2;
				double u1x = AmrQGD::u1x,
				u2x = AmrQGD::u2x,
				u1y = AmrQGD::u1y,
				u2y = AmrQGD::u2y,
				u1z = AmrQGD::u1z,
				u2z = AmrQGD::u2z;
				double E_in01 = AmrQGD::E_in01, E_in02 = AmrQGD::E_in02;//0.0; //y_1 = 0; *y_2 = 0.0;
				double esp = AmrQGD::esp;
				double x_bub = AmrQGD::x_bub;
				double r_bub = AmrQGD::r_bub;
			//}
			
            //surfaces
            if (i < ilo) {// Left bound
                dest(i,j,k,9) = dest(ilo,j,k,iE_in0); 			 										//E_in0
                dest(i,j,k,11) = dest(ilo,j,k,iCs); 			 										//Cs
                dest(i,j,k,12) = 1 - esp; 						 										//Va
                dest(i,j,k,13) = 1 - dest(i,j,k,iVa); 			 										//Vb
				dest(i,j,k,3) = u1x; 							 										//ux
                dest(i,j,k,4) = u1y; 							 										//uy
                dest(i,j,k,5) = u1z; 							 										//uz
				dest(i,j,k,10) = T1; 							 										//T
				dest(i,j,k,6) = dest(ilo,j,k,ip); 				 										//p
				dest(i,j,k,1) = dest(i,j,k,iVa) * (dest(i,j,k,ip) + pa_inf) / (Ra * dest(i,j,k,iT));	//rho_a
                dest(i,j,k,2) = dest(i,j,k,iVb) * (dest(i,j,k,ip) + pb_inf) / (Rb * dest(i,j,k,iT));	//rho_b
				dest(i,j,k,0) = dest(i,j,k,ira)+dest(i,j,k,irb); 										//rho
				double E_in = cva*dest(i,j,k,iT) + dest(i,j,k,iT) * Ra * pa_inf / (dest(i,j,k,ip) + pa_inf) + dest(i,j,k,iE_in0);// [2](9)
                E_in = dest(i,j,k,ira)*E_in + dest(i,j,k,irb)*(cvb*dest(i,j,k,iT) + dest(i,j,k,iT) * Rb * pb_inf / (dest(i,j,k,ip) + pb_inf) + dest(i,j,k,iE_in0));
				dest(i,j,k,8) = E_in/dest(i,j,k,ir); 			 										//E_in
				double uu = (dest(i,j,k,iux)*dest(i,j,k,iux)+dest(i,j,k,iuy)*dest(i,j,k,iuy)+dest(i,j,k,iuz)*dest(i,j,k,iuz));
                dest(i,j,k,7) = dest(i,j,k,iE_in)*dest(i,j,k,ir)+0.5*dest(i,j,k,ir)*uu; 				//E
                
            }
            if (i > ihi) {// Right bound
                dest(i,j,k,0) = dest(ihi,j,k,ir);
                dest(i,j,k,1) = dest(ihi,j,k,1);
                dest(i,j,k,2) = dest(ihi,j,k,2);
                dest(i,j,k,3) = dest(ihi,j,k,iux);
                dest(i,j,k,4) = dest(ihi,j,k,iuy);
                dest(i,j,k,5) = dest(ihi,j,k,iuz);
                dest(i,j,k,6) = pR;
				dest(i,j,k,8) = dest(ihi,j,k,iE_in);
				double uu = (dest(i,j,k,iux)*dest(i,j,k,iux)+dest(i,j,k,iuy)*dest(i,j,k,iuy)+dest(i,j,k,iuz)*dest(i,j,k,iuz));
                dest(i,j,k,7) = dest(i,j,k,ir)*(uu)*0.5+dest(i,j,k,ir)*dest(i,j,k,iE_in);
                dest(i,j,k,9) = dest(ihi,j,k,9);
                dest(i,j,k,10) = dest(ihi,j,k,10);
                dest(i,j,k,11) = dest(ihi,j,k,11);
                dest(i,j,k,12) = dest(ihi,j,k,12);
                dest(i,j,k,13) = dest(ihi,j,k,13);
            }
            if (j < jlo) {// Front bound
                dest(i,j,k,0) = dest(i,jlo,k,0);
                dest(i,j,k,1) = dest(i,jlo,k,1);
                dest(i,j,k,2) = dest(i,jlo,k,2);
                dest(i,j,k,3) = dest(i,jlo,k,3);
                dest(i,j,k,4) = 0;//dest(i,jlo,k,iuy);//-dest(i,jlo,k,iuy); //- if y from 0, not -15e-03
                dest(i,j,k,5) = dest(i,jlo,k,5);
                dest(i,j,k,6) = dest(i,jlo,k,6);
                dest(i,j,k,7) = dest(i,jlo,k,7);
                dest(i,j,k,8) = dest(i,jlo,k,8);
                dest(i,j,k,9) = dest(i,jlo,k,9);
                dest(i,j,k,10) = dest(i,jlo,k,10);
                dest(i,j,k,11) = dest(i,jlo,k,11);
                dest(i,j,k,12) = dest(i,jlo,k,12);
                dest(i,j,k,13) = dest(i,jlo,k,13);
            }
            if (j > jhi) {// Rear bound
                dest(i,j,k,0) = dest(i,jhi,k,0);
                dest(i,j,k,1) = dest(i,jhi,k,1);
                dest(i,j,k,2) = dest(i,jhi,k,2);
                dest(i,j,k,3) = dest(i,jhi,k,3);
                dest(i,j,k,4) = dest(i,jhi,k,4);
                dest(i,j,k,5) = dest(i,jhi,k,5);
                dest(i,j,k,6) = dest(i,jhi,k,6);
                dest(i,j,k,7) = dest(i,jhi,k,7);
                dest(i,j,k,8) = dest(i,jhi,k,8);
                dest(i,j,k,9) = dest(i,jhi,k,9);
                dest(i,j,k,10) = dest(i,jhi,k,10);
                dest(i,j,k,11) = dest(i,jhi,k,11);
                dest(i,j,k,12) = dest(i,jhi,k,12);
                dest(i,j,k,13) = dest(i,jhi,k,13);
            }
            if (k < klo) {// Lower bound .no minus => set periodic conditions later
                dest(i,j,k,0) = dest(i,j,klo,0);
                dest(i,j,k,1) = dest(i,j,klo,1);
                dest(i,j,k,2) = dest(i,j,klo,2);
                dest(i,j,k,3) = dest(i,j,klo,3);
                dest(i,j,k,4) = dest(i,j,klo,4);
                dest(i,j,k,iuz) = dest(i,j,klo,iuz);//0;//dest(i,j,klo,iuz);
                dest(i,j,k,6) = dest(i,j,klo,6);
                dest(i,j,k,7) = dest(i,j,klo,7);
                dest(i,j,k,8) = dest(i,j,klo,8);
                dest(i,j,k,9) = dest(i,j,klo,9);
                dest(i,j,k,10) = dest(i,j,klo,10);
                dest(i,j,k,11) = dest(i,j,klo,11);
                dest(i,j,k,12) = dest(i,j,klo,12);
                dest(i,j,k,13) = dest(i,j,klo,13);/*
                dest(i,j,k,0) = dest(i,j,khi,0);
                dest(i,j,k,1) = dest(i,j,khi,1);
                dest(i,j,k,2) = dest(i,j,khi,2);
                dest(i,j,k,3) = dest(i,j,khi,3);
                dest(i,j,k,4) = dest(i,j,khi,4);
                dest(i,j,k,5) = dest(i,j,khi,5);
                dest(i,j,k,6) = dest(i,j,khi,6);
                dest(i,j,k,7) = dest(i,j,khi,7);
                dest(i,j,k,8) = dest(i,j,khi,8);
                dest(i,j,k,9) = dest(i,j,khi,9);
                dest(i,j,k,10) = dest(i,j,khi,10);
                dest(i,j,k,11) = dest(i,j,khi,11);
                dest(i,j,k,12) = dest(i,j,khi,12);
                dest(i,j,k,13) = dest(i,j,khi,13);*/
				//std::cout << "fillBC.cpp2 k=" << k <<" klo="<< klo << " \n";
				//std::cout << "fillBC.cpp (" << i << ", "<< j << ", "<< k << ") \n";
				//std::cout << "fillBC.cpp2 uz=" << dest(i,j,k,5) << " \n";
            }
            if (k > khi) {// Upper bound
                dest(i,j,k,0) = dest(i,j,khi,0);
                dest(i,j,k,1) = dest(i,j,khi,1);
                dest(i,j,k,2) = dest(i,j,khi,2);
                dest(i,j,k,3) = dest(i,j,khi,3);
                dest(i,j,k,4) = dest(i,j,khi,4);
                dest(i,j,k,5) = dest(i,j,khi,5);
                dest(i,j,k,6) = dest(i,j,khi,6);
                dest(i,j,k,7) = dest(i,j,khi,7);
                dest(i,j,k,8) = dest(i,j,khi,8);
                dest(i,j,k,9) = dest(i,j,khi,9);
                dest(i,j,k,10) = dest(i,j,khi,10);
                dest(i,j,k,11) = dest(i,j,khi,11);
                dest(i,j,k,12) = dest(i,j,khi,12);
                dest(i,j,k,13) = dest(i,j,khi,13);/*
                dest(i,j,k,0) = dest(i,j,klo,0);
                dest(i,j,k,1) = dest(i,j,klo,1);
                dest(i,j,k,2) = dest(i,j,klo,2);
                dest(i,j,k,3) = dest(i,j,klo,3);
                dest(i,j,k,4) = dest(i,j,klo,4);
                dest(i,j,k,5) = dest(i,j,klo,5);
                dest(i,j,k,6) = dest(i,j,klo,6);
                dest(i,j,k,7) = dest(i,j,klo,7);
                dest(i,j,k,8) = dest(i,j,klo,8);
                dest(i,j,k,9) = dest(i,j,klo,9);
                dest(i,j,k,10) = dest(i,j,klo,10);
                dest(i,j,k,11) = dest(i,j,klo,11);
                dest(i,j,k,12) = dest(i,j,klo,12);
                dest(i,j,k,13) = dest(i,j,klo,13);*/
            }
            
            //-------------edges--------------//
            /**/
            if(i < ilo && j < jlo) {//left mirror into, without j
                dest(i,j,k,0) = dest(ilo,jlo,k,0);
                dest(i,j,k,1) = dest(ilo,jlo,k,1);
                dest(i,j,k,2) = dest(ilo,jlo,k,2);
                dest(i,j,k,3) = dest(ilo,jlo,k,3);
                dest(i,j,k,4) = dest(ilo,jlo,k,4);
                dest(i,j,k,5) = dest(ilo,jlo,k,5);
                dest(i,j,k,6) = dest(ilo,jlo,k,6);
                dest(i,j,k,7) = dest(ilo,jlo,k,7);
                dest(i,j,k,8) = dest(ilo,jlo,k,8);
                dest(i,j,k,9) = dest(ilo,jlo,k,9);   
                dest(i,j,k,10) = dest(ilo,jlo,k,10);
                dest(i,j,k,11) = dest(ilo,jlo,k,11);
                dest(i,j,k,12) = dest(ilo,jlo,k,12);
                dest(i,j,k,13) = dest(ilo,jlo,k,13);     
            }
            if(i < ilo && j > jhi) {
                dest(i,j,k,0) = dest(ilo,jhi,k,0);
                dest(i,j,k,1) = dest(ilo,jhi,k,1);
                dest(i,j,k,2) = dest(ilo,jhi,k,2);
                dest(i,j,k,3) = dest(ilo,jhi,k,3);
                dest(i,j,k,4) = dest(ilo,jhi,k,4);
                dest(i,j,k,5) = dest(ilo,jhi,k,5);
                dest(i,j,k,6) = dest(ilo,jhi,k,6);
                dest(i,j,k,7) = dest(ilo,jhi,k,7);
                dest(i,j,k,8) = dest(ilo,jhi,k,8);
                dest(i,j,k,9) = dest(ilo,jhi,k,9);  
                dest(i,j,k,10) = dest(ilo,jhi,k,10);
                dest(i,j,k,11) = dest(ilo,jhi,k,11);
                dest(i,j,k,12) = dest(ilo,jhi,k,12);
                dest(i,j,k,13) = dest(ilo,jhi,k,13);         
            }                        
            if(i < ilo && k < klo) {
                dest(i,j,k,0) = dest(ilo,j,klo,0);
                dest(i,j,k,1) = dest(ilo,j,klo,1);
                dest(i,j,k,2) = dest(ilo,j,klo,2);
                dest(i,j,k,3) = dest(ilo,j,klo,3);
                dest(i,j,k,4) = dest(ilo,j,klo,4);
                dest(i,j,k,5) = dest(ilo,j,klo,5);
                dest(i,j,k,6) = dest(ilo,j,klo,6);
                dest(i,j,k,7) = dest(ilo,j,klo,7);
                dest(i,j,k,8) = dest(ilo,j,klo,8);
                dest(i,j,k,9) = dest(ilo,j,klo,9);
                dest(i,j,k,10) = dest(ilo,j,klo,10);
                dest(i,j,k,11) = dest(ilo,j,klo,11);
                dest(i,j,k,12) = dest(ilo,j,klo,12);
                dest(i,j,k,13) = dest(ilo,j,klo,13);
            }            
            if(i < ilo && k > khi) {
                dest(i,j,k,0) = dest(ilo,j,khi,0);
                dest(i,j,k,1) = dest(ilo,j,khi,1);
                dest(i,j,k,2) = dest(ilo,j,khi,2);
                dest(i,j,k,3) = dest(ilo,j,khi,3);
                dest(i,j,k,4) = dest(ilo,j,khi,4);
                dest(i,j,k,5) = dest(ilo,j,khi,5);
                dest(i,j,k,6) = dest(ilo,j,khi,6);
                dest(i,j,k,7) = dest(ilo,j,khi,7);
                dest(i,j,k,8) = dest(ilo,j,khi,8);
                dest(i,j,k,9) = dest(ilo,j,khi,9);
                dest(i,j,k,10) = dest(ilo,j,khi,10);
                dest(i,j,k,11) = dest(ilo,j,khi,11);
                dest(i,j,k,12) = dest(ilo,j,khi,12);
                dest(i,j,k,13) = dest(ilo,j,khi,13);
            }            
            if(i > ihi && j < jlo) {
                dest(i,j,k,0) = dest(ihi,jlo,k,0);
                dest(i,j,k,1) = dest(ihi,jlo,k,1);
                dest(i,j,k,2) = dest(ihi,jlo,k,2);
                dest(i,j,k,3) = dest(ihi,jlo,k,3);
                dest(i,j,k,4) = dest(ihi,jlo,k,4);
                dest(i,j,k,5) = dest(ihi,jlo,k,5);
                dest(i,j,k,6) = dest(ihi,jlo,k,6);
                dest(i,j,k,7) = dest(ihi,jlo,k,7);
                dest(i,j,k,8) = dest(ihi,jlo,k,8);
                dest(i,j,k,9) = dest(ihi,jlo,k,9);  
                dest(i,j,k,10) = dest(ihi,jlo,k,10);
                dest(i,j,k,11) = dest(ihi,jlo,k,11);
                dest(i,j,k,12) = dest(ihi,jlo,k,12);
                dest(i,j,k,13) = dest(ihi,jlo,k,13);       
            }
            if(i > ihi && j > jhi) {
                dest(i,j,k,0) = dest(ihi,jhi,k,0);
                dest(i,j,k,1) = dest(ihi,jhi,k,1);
                dest(i,j,k,2) = dest(ihi,jhi,k,2);
                dest(i,j,k,3) = dest(ihi,jhi,k,3);
                dest(i,j,k,4) = dest(ihi,jhi,k,4);
                dest(i,j,k,5) = dest(ihi,jhi,k,5);
                dest(i,j,k,6) = dest(ihi,jhi,k,6);
                dest(i,j,k,7) = dest(ihi,jhi,k,7);
                dest(i,j,k,8) = dest(ihi,jhi,k,8);
                dest(i,j,k,9) = dest(ihi,jhi,k,9);  
                dest(i,j,k,10) = dest(ihi,jhi,k,10);
                dest(i,j,k,11) = dest(ihi,jhi,k,11);
                dest(i,j,k,12) = dest(ihi,jhi,k,12);
                dest(i,j,k,13) = dest(ihi,jhi,k,13);         
            }                        
            if(i > ihi && k < klo) {
                dest(i,j,k,0) = dest(ihi,j,klo,0);
                dest(i,j,k,1) = dest(ihi,j,klo,1);
                dest(i,j,k,2) = dest(ihi,j,klo,2);
                dest(i,j,k,3) = dest(ihi,j,klo,3);
                dest(i,j,k,4) = dest(ihi,j,klo,4);
                dest(i,j,k,5) = dest(ihi,j,klo,5);
                dest(i,j,k,6) = dest(ihi,j,klo,6);
                dest(i,j,k,7) = dest(ihi,j,klo,7);
                dest(i,j,k,8) = dest(ihi,j,klo,8);
                dest(i,j,k,9) = dest(ihi,j,klo,9);
                dest(i,j,k,10) = dest(ihi,j,klo,10);
                dest(i,j,k,11) = dest(ihi,j,klo,11);
                dest(i,j,k,12) = dest(ihi,j,klo,12);
                dest(i,j,k,13) = dest(ihi,j,klo,13);
            }            
            if(i > ihi && k > khi) {
                dest(i,j,k,0) = dest(ihi,j,klo,0);
                dest(i,j,k,1) = dest(ihi,j,klo,1);
                dest(i,j,k,2) = dest(ihi,j,klo,2);
                dest(i,j,k,3) = dest(ihi,j,klo,3);
                dest(i,j,k,4) = dest(ihi,j,klo,4);
                dest(i,j,k,5) = dest(ihi,j,klo,5);
                dest(i,j,k,6) = dest(ihi,j,klo,6);
                dest(i,j,k,7) = dest(ihi,j,klo,7);
                dest(i,j,k,8) = dest(ihi,j,klo,8);
                dest(i,j,k,9) = dest(ihi,j,klo,9);
                dest(i,j,k,10) = dest(ihi,j,klo,10);
                dest(i,j,k,11) = dest(ihi,j,klo,11);
                dest(i,j,k,12) = dest(ihi,j,klo,12);
                dest(i,j,k,13) = dest(ihi,j,klo,13);
            }
                         
            if(j < jlo && k < klo) {
                dest(i,j,k,0) = dest(i,jlo,klo,0);
                dest(i,j,k,1) = dest(i,jlo,klo,1);
                dest(i,j,k,2) = dest(i,jlo,klo,2);
                dest(i,j,k,3) = dest(i,jlo,klo,3);
                dest(i,j,k,4) = dest(i,jlo,klo,iuy);
                dest(i,j,k,5) = dest(i,jlo,klo,5);
                dest(i,j,k,6) = dest(i,jlo,klo,6);
                dest(i,j,k,7) = dest(i,jlo,klo,7);
                dest(i,j,k,8) = dest(i,jlo,klo,8);
                dest(i,j,k,9) = dest(i,jlo,klo,9);
                dest(i,j,k,10) = dest(i,jlo,klo,10);
                dest(i,j,k,11) = dest(i,jlo,klo,11);
                dest(i,j,k,12) = dest(i,jlo,klo,12);
                dest(i,j,k,13) = dest(i,jlo,klo,13);
            }
            if(j < jlo && k > khi) {
                dest(i,j,k,0) = dest(i,jlo,khi,0);
                dest(i,j,k,1) = dest(i,jlo,khi,1);
                dest(i,j,k,2) = dest(i,jlo,khi,2);
                dest(i,j,k,3) = dest(i,jlo,khi,3);
                dest(i,j,k,4) = dest(i,jlo,khi,iuy);
                dest(i,j,k,5) = dest(i,jlo,khi,5);
                dest(i,j,k,6) = dest(i,jlo,khi,6);
                dest(i,j,k,7) = dest(i,jlo,khi,7);
                dest(i,j,k,8) = dest(i,jlo,khi,8);
                dest(i,j,k,9) = dest(i,jlo,khi,9);
                dest(i,j,k,10) = dest(i,jlo,khi,10);
                dest(i,j,k,11) = dest(i,jlo,khi,11);
                dest(i,j,k,12) = dest(i,jlo,khi,12);
                dest(i,j,k,13) = dest(i,jlo,khi,13);
            }            
            if(j > jhi && k < klo) {
                dest(i,j,k,0) = dest(i,jhi,klo,0);
                dest(i,j,k,1) = dest(i,jhi,klo,1);
                dest(i,j,k,2) = dest(i,jhi,klo,2);
                dest(i,j,k,3) = dest(i,jhi,klo,3);
                dest(i,j,k,4) = dest(i,jhi,klo,4);
                dest(i,j,k,5) = dest(i,jhi,klo,5);
                dest(i,j,k,6) = dest(i,jhi,klo,6);
                dest(i,j,k,7) = dest(i,jhi,klo,7);
                dest(i,j,k,8) = dest(i,jhi,klo,8);
                dest(i,j,k,9) = dest(i,jhi,klo,9);
                dest(i,j,k,10) = dest(i,jhi,klo,10);
                dest(i,j,k,11) = dest(i,jhi,klo,11);
                dest(i,j,k,12) = dest(i,jhi,klo,12);
                dest(i,j,k,13) = dest(i,jhi,klo,13);
            }
            if(j > jhi && k > khi) {
                dest(i,j,k,0) = dest(i,jhi,khi,0);
                dest(i,j,k,1) = dest(i,jhi,khi,1);
                dest(i,j,k,2) = dest(i,jhi,khi,2);
                dest(i,j,k,3) = dest(i,jhi,khi,3);
                dest(i,j,k,4) = dest(i,jhi,khi,4);
                dest(i,j,k,5) = dest(i,jhi,khi,5);
                dest(i,j,k,6) = dest(i,jhi,khi,6);
                dest(i,j,k,7) = dest(i,jhi,khi,7);
                dest(i,j,k,8) = dest(i,jhi,khi,8);
                dest(i,j,k,9) = dest(i,jhi,khi,9);
                dest(i,j,k,10) = dest(i,jhi,khi,10);
                dest(i,j,k,11) = dest(i,jhi,khi,11);
                dest(i,j,k,12) = dest(i,jhi,khi,12);
                dest(i,j,k,13) = dest(i,jhi,khi,13);
            }  
                       
            //----------corner cells----------//
            //CC: not included in the calculation. But it will be in mixed derivatives. Use a half-sum
            if(i < ilo && j < jlo && k < klo) {                
                dest(i,j,k,0) = dest(ilo,jlo,klo,0);
                dest(i,j,k,1) = dest(ilo,jlo,klo,1);
                dest(i,j,k,2) = dest(ilo,jlo,klo,2);
                dest(i,j,k,3) = dest(ilo,jlo,klo,3);
                dest(i,j,k,4) = dest(ilo,jlo,klo,4);
                dest(i,j,k,5) = dest(ilo,jlo,klo,5);
                dest(i,j,k,6) = dest(ilo,jlo,klo,6);
                dest(i,j,k,7) = dest(ilo,jlo,klo,7);
                dest(i,j,k,8) = dest(ilo,jlo,klo,8);
                dest(i,j,k,9) = dest(ilo,jlo,klo,9);
                dest(i,j,k,10) = dest(ilo,jlo,klo,10);
                dest(i,j,k,11) = dest(ilo,jlo,klo,11);
                dest(i,j,k,12) = dest(ilo,jlo,klo,12);
                dest(i,j,k,13) = dest(ilo,jlo,klo,13);
            }
            if(i < ilo && j < jlo && k > khi) {
                dest(i,j,k,0) = dest(ilo,jlo,khi,0);
                dest(i,j,k,1) = dest(ilo,jlo,khi,1);
                dest(i,j,k,2) = dest(ilo,jlo,khi,2);
                dest(i,j,k,3) = dest(ilo,jlo,khi,3);
                dest(i,j,k,4) = dest(ilo,jlo,khi,4);
                dest(i,j,k,5) = dest(ilo,jlo,khi,5);
                dest(i,j,k,6) = dest(ilo,jlo,khi,6);
                dest(i,j,k,7) = dest(ilo,jlo,khi,7);
                dest(i,j,k,8) = dest(ilo,jlo,khi,8);
                dest(i,j,k,9) = dest(ilo,jlo,khi,9);
                dest(i,j,k,10) = dest(ilo,jlo,khi,10);
                dest(i,j,k,11) = dest(ilo,jlo,khi,11);
                dest(i,j,k,12) = dest(ilo,jlo,khi,12);
                dest(i,j,k,13) = dest(ilo,jlo,khi,13);
            }
            
            if(i < ilo && j > jhi && k < klo) {
                dest(i,j,k,0) = dest(ilo,jhi,klo,0);
                dest(i,j,k,1) = dest(ilo,jhi,klo,1);
                dest(i,j,k,2) = dest(ilo,jhi,klo,2);
                dest(i,j,k,3) = dest(ilo,jhi,klo,3);
                dest(i,j,k,4) = dest(ilo,jhi,klo,4);
                dest(i,j,k,5) = dest(ilo,jhi,klo,5);
                dest(i,j,k,6) = dest(ilo,jhi,klo,6);
                dest(i,j,k,7) = dest(ilo,jhi,klo,7);
                dest(i,j,k,8) = dest(ilo,jhi,klo,8);
                dest(i,j,k,9) = dest(ilo,jhi,klo,9);
                dest(i,j,k,10) = dest(ilo,jhi,klo,10);
                dest(i,j,k,11) = dest(ilo,jhi,klo,11);
                dest(i,j,k,12) = dest(ilo,jhi,klo,12);
                dest(i,j,k,13) = dest(ilo,jhi,klo,13);
            }
            if(i < ilo && j > jhi && k > khi) {
                dest(i,j,k,0) = dest(ilo,jhi,khi,0);
                dest(i,j,k,1) = dest(ilo,jhi,khi,1);
                dest(i,j,k,2) = dest(ilo,jhi,khi,2);
                dest(i,j,k,3) = dest(ilo,jhi,khi,3);
                dest(i,j,k,4) = dest(ilo,jhi,khi,4);
                dest(i,j,k,5) = dest(ilo,jhi,khi,5);
                dest(i,j,k,6) = dest(ilo,jhi,khi,6);
                dest(i,j,k,7) = dest(ilo,jhi,khi,7);
                dest(i,j,k,8) = dest(ilo,jhi,khi,8);
                dest(i,j,k,9) = dest(ilo,jhi,khi,9);
                dest(i,j,k,10) = dest(ilo,jhi,khi,10);
                dest(i,j,k,11) = dest(ilo,jhi,khi,11);
                dest(i,j,k,12) = dest(ilo,jhi,khi,12);
                dest(i,j,k,13) = dest(ilo,jhi,khi,13);
            }
            
            if(i > ihi && j < jlo && k < klo) {
                dest(i,j,k,0) = dest(ihi,jlo,klo,0);
                dest(i,j,k,1) = dest(ihi,jlo,klo,1);
                dest(i,j,k,2) = dest(ihi,jlo,klo,2);
                dest(i,j,k,3) = dest(ihi,jlo,klo,3);
                dest(i,j,k,4) = dest(ihi,jlo,klo,4);
                dest(i,j,k,5) = dest(ihi,jlo,klo,5);
                dest(i,j,k,6) = dest(ihi,jlo,klo,6);
                dest(i,j,k,7) = dest(ihi,jlo,klo,7);
                dest(i,j,k,8) = dest(ihi,jlo,klo,8);
                dest(i,j,k,9) = dest(ihi,jlo,klo,9);
                dest(i,j,k,10) = dest(ihi,jlo,klo,10);
                dest(i,j,k,11) = dest(ihi,jlo,klo,11);
                dest(i,j,k,12) = dest(ihi,jlo,klo,12);
                dest(i,j,k,13) = dest(ihi,jlo,klo,13);
            }
            if(i > ihi && j < jlo && k > khi) {
                dest(i,j,k,0) = dest(ihi,jlo,khi,0);
                dest(i,j,k,1) = dest(ihi,jlo,khi,1);
                dest(i,j,k,2) = dest(ihi,jlo,khi,2);
                dest(i,j,k,3) = dest(ihi,jlo,khi,3);
                dest(i,j,k,4) = dest(ihi,jlo,khi,4);
                dest(i,j,k,5) = dest(ihi,jlo,khi,5);
                dest(i,j,k,6) = dest(ihi,jlo,khi,6);
                dest(i,j,k,7) = dest(ihi,jlo,khi,7);
                dest(i,j,k,8) = dest(ihi,jlo,khi,8);
                dest(i,j,k,9) = dest(ihi,jlo,khi,9);
                dest(i,j,k,10) = dest(ihi,jlo,khi,10);
                dest(i,j,k,11) = dest(ihi,jlo,khi,11);
                dest(i,j,k,12) = dest(ihi,jlo,khi,12);
                dest(i,j,k,13) = dest(ihi,jlo,khi,13);
            }
            
            if(i > ihi && j > jhi && k < klo) {
                dest(i,j,k,0) = dest(ihi,jhi,klo,0);
                dest(i,j,k,1) = dest(ihi,jhi,klo,1);
                dest(i,j,k,2) = dest(ihi,jhi,klo,2);
                dest(i,j,k,3) = dest(ihi,jhi,klo,3);
                dest(i,j,k,4) = dest(ihi,jhi,klo,4);
                dest(i,j,k,5) = dest(ihi,jhi,klo,5);
                dest(i,j,k,6) = dest(ihi,jhi,klo,6);
                dest(i,j,k,7) = dest(ihi,jhi,klo,7);
                dest(i,j,k,8) = dest(ihi,jhi,klo,8);
                dest(i,j,k,9) = dest(ihi,jhi,klo,9);
                dest(i,j,k,10) = dest(ihi,jhi,klo,10);
                dest(i,j,k,11) = dest(ihi,jhi,klo,11);
                dest(i,j,k,12) = dest(ihi,jhi,klo,12);
                dest(i,j,k,13) = dest(ihi,jhi,klo,13);
            }
            if(i > ihi && j > jhi && k > khi) {
                dest(i,j,k,0) = dest(ihi,jhi,khi,0);
                dest(i,j,k,1) = dest(ihi,jhi,khi,1);
                dest(i,j,k,2) = dest(ihi,jhi,khi,2);
                dest(i,j,k,3) = dest(ihi,jhi,khi,3);
                dest(i,j,k,4) = dest(ihi,jhi,khi,4);
                dest(i,j,k,5) = dest(ihi,jhi,khi,5);
                dest(i,j,k,6) = dest(ihi,jhi,khi,6);
                dest(i,j,k,7) = dest(ihi,jhi,khi,7);
                dest(i,j,k,8) = dest(ihi,jhi,khi,8);
                dest(i,j,k,9) = dest(ihi,jhi,khi,9);
                dest(i,j,k,10) = dest(ihi,jhi,khi,10);
                dest(i,j,k,11) = dest(ihi,jhi,khi,11);
                dest(i,j,k,12) = dest(ihi,jhi,khi,12);
                dest(i,j,k,13) = dest(ihi,jhi,khi,13);
            }
			//dest(i,j,k,iuz) = 0;
		/* */	
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
