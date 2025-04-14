#include "AmrQGD.H"
#include <iostream>

using namespace amrex;

Real AmrQGD::advance (Real time, Real dt, int iteration, int ncycle)
{
	// At the beginning of step, we make the new data from previous step the
	// old data of this step.
	for (int k = 0; k < NUM_STATE_TYPE; ++k) {
		state[k].allocOldData();
		state[k].swapTimeLevels(dt);
	}

	double mu_T = mutGas;
    
	auto dx = Geom().CellSizeArray();
	
	auto hx_i = dx[0]; 
	auto hy_j = dx[1]; 
	auto hz_k = dx[2]; 
	
	MultiFab& S_new = state[0].newData();
	auto const& VNew = S_new.arrays();
	MultiFab& S_old = state[0].oldData();
	FillPatcherFill(S_old, 0, ncomp, nghost, time, State_Type, 0);

	auto const& VOld = S_old.arrays();

	double maxCs = 0.;
	
	int ir = 0, ira = 1, irb = 2, 
	iux = 3, iuy = 4, iuz = 5,
	ip = 6, iE = 7, iE_in = 8, iE_in0 = 9,
	iT = 10, 
	iCs = 11,  //or auto const& Cs = S_new[0].arrays();,  
	iVa = 12, iVb = 13;
	
	double alpha = alphaQgd;
	
	double Sc = ScQgd, Pr = PrQgd;
	double Fx = 0.0, Fy = 0.0, Fz = 0.0;
	double Q = 0.0;
	double gma = gamma_a;
	double gmb = gamma_b;

	double Ra = RGas_a;
	double Rb = RGas_b;
	double cva = Ra/(gma-1);//717.5 Air
	double cvb = Rb/(gmb-1);//1495 Water
 
	double cpa = cva*gma;//Air 1004.5
	double cpb = cvb*gmb;//Kitamura: Water 4186
	//Ra = cpa*(gma - 1)/gma
	//Rb = cpb*(gmb - 1)/gmb; 
	
	double pa_inf = painf, pb_inf = pbinf;
	
	
	/*//{
		int nc1_x, nc1_y, nc1_z;//+3
		//nc1_x = 700, nc1_y = 400, nc1_z = 8;
		amrex::Box box = S_new.boxArray()[0];
		nc1_x = box.length(0), nc1_y = box.length(1), nc1_z = box.length(2);
		int nc_x = nc1_x-1, nc_y = nc1_y-1, nc_z = nc1_z-1;
		int nil = 0;//zero
		int bi = 0;
		int ze = 0;// -2
		//amrex::Print() << "\n NumOfArrs nComp = " << S_new.nComp();
		//amrex::Print() << "\n NumOfArrs ncomp = " << ncomp;
		//amrex::Print() << "\n Nx = " << box.length(0);
		//amrex::Print() << " Ny = " << box.length(1);
		//mrex::Print() << " Nz = " << box.length(2) << "\n";
		nil = -1;
		nc_x = nc1_x, nc_y = nc1_y, nc_z = nc1_z;
	//}*/
	//int i=-1, j=0, k =0;
	//amrex::Print() << ", " << VOld[bi](i,j,k,ip) << " vs "; amrex::Print()  << VNew[bi](i,j,k,ip); amrex::Print()  << " (i=" << i << ") \n";
	
	//S_new = {0};
	//-VNew[0] = {0};
	amrex::ParallelFor(S_old, [=] AMREX_GPU_DEVICE (int bi, int i, int j, int k)
	{
		for (int ia=0; ia<ncomp; ia++){//for (int ip=0; ip<16; ip++)
			VNew[bi](i,j,k,ia) = 0.0;
			//amrex::Print() << ", ia= " << ia << "\n";//14=0..13
			//-VNew[bi](i-1,j,k,ia) = 0.0;
			/*VNew[bi](i+1,j,k,ia) = 0.0;//->if i=endI for endI+1 
			//-VNew[bi](i,j-1,k,ia) = 0.0;
			VNew[bi](i,j+1,k,ia) = 0.0;//if j>ncx
			//-VNew[bi](i,j,k-1,ia) = 0.0;
			VNew[bi](i,j,k+1,ia) = 0.0;//find way for all 'ParallelFor' (w bnd)
			//-VNew[bi](i,j,k-2,ia) = 0.0;VNew[bi](i,j,k+2,ia) = 0.0;
			*/
		}
		//if
			//amrex::Print() << ", " << VNew[bi](i,j,-1,iuz) << " (i=" << i << ") \n";
			//amrex::Print() << ", " << VNew[bi](i,j,4+1,iuz) << " (i=" << i << ") \n";
	});
	
	//%% Time step
	//% Calculation dt
	if (typeCs == 1) {
		// typeCs = 1 - according to Zlotnik's seminar, it is more correct
		// [new2].(33)
		amrex::ParallelFor(S_old, [=, &maxCs] AMREX_GPU_DEVICE (int bi, int i, int j, int k)
		{
			double ro = VOld[bi](i,j,k,ir);
			double roa = VOld[bi](i,j,k,ira); double rob = VOld[bi](i,j,k,irb);
			double p = VOld[bi](i,j,k,ip);
	
			double adE_in = VOld[bi](i,j,k,iE_in) - VOld[bi](i,j,k,iE_in0);
			double rm = VOld[bi](i,j,k,ir), rma = VOld[bi](i,j,k,ira), rmb = VOld[bi](i,j,k,irb); // It's rmaf
			double cvm = (rma * cva + rmb * cvb) / rm;
			double cpm = (rma * cpa + rmb * cpb) / rm;
			double gam = cpm / cvm;
		    double sigma_a = Ra * roa / (cvm * ro);//% 2.(18)
			double sigma_b = Rb * rob / (cvm * ro);
			// p is p_+
			double E1 = sigma_a * (ro * adE_in - pa_inf) / pow(p + pa_inf, 2.); 
			E1 = E1 + sigma_b * (ro * adE_in - pb_inf) / pow(p + pb_inf, 2.); 
			E1 = 1.0 / E1; 
			E1 = sqrt(gam / ro * E1); 
			double Cs = E1;
			VNew[bi](i,j,k,iCs) = Cs;
			if (isnan(VNew[bi](i,j,k,iCs)) || isinf(VNew[bi](i,j,k,iCs)) || VNew[bi](i,j,k,iCs) <= 0 )
			{
				VNew[bi](i,j,k,iCs) = 1.0 * pow(10, -8);
				Cs = VNew[bi](i,j,k,iCs);
				exit(EXIT_FAILURE);
			}
			if (Cs > maxCs) 
			{
				maxCs = Cs;
			}
		});
	}
	else if (typeCs == 100) {
		// H_2D считает, но не долго.Лучше самую первую Cs
		// А вот I_2D на стеках считает
		amrex::ParallelFor(S_old, [=, &maxCs] AMREX_GPU_DEVICE (int bi, int i, int j, int k)
		{
			double rma = VOld[bi](i,j,k,ira), rmb = VOld[bi](i,j,k,irb);
			double gamCs = gma * rma * Ra + gmb * rmb * Rb;
			double Tm = VOld[bi](i,j,k,iT); double rm = VOld[bi](i,j,k,ir);
			double Cs = sqrt(gamCs * Tm / rm); // H_2D(A_3D-waterCI) не считает, а I_2D(B_3D-airCI) - считает.
			VNew[bi](i,j,k,iCs) = Cs;
				
			if (isnan(VNew[bi](i,j,k,iCs)) || isinf(VNew[bi](i,j,k,iCs)) || VNew[bi](i,j,k,iCs) <= 0 )
			{
				fprintf(stderr, "\niCs100 %d,%d,%d Cs = %le "
					"v1=%le v2=%f v3=%le", i,j,k, Cs, 
					rma, rmb, rm);
			
				VNew[bi](i,j,k,iCs) = 1.0 * pow(10, -8);//test5
				//-VNew[bi](i,j,k,iCs) = VNew[bi](i-1,j-1,k-1,iCs);
				Cs = VNew[bi](i,j,k,iCs);
				exit(EXIT_FAILURE);
			}
			if (Cs > maxCs) {
				maxCs = Cs;
			}
		});
	}
	else if (typeCs == 101) {
		// almost the same as 100.
		// 7 page, after formula(19) % On Regularized Systems of Equations for Gas
		// Mixture Dynamics with New Regularizing Velocities and Diffusion Fluxes
		amrex::ParallelFor(S_old, [=, &maxCs] AMREX_GPU_DEVICE (int bi, int i, int j, int k)
		{
			double rm = VOld[bi](i,j,k,ir), rma = VOld[bi](i,j,k,ira), rmb = VOld[bi](i,j,k,irb);
			double cvm = (rma * cva + rmb * cvb) / rm; double cpm = (rma * cpa + rmb * cpb) / rm;
			double gam = cpm / cvm; double R = (rma * Ra + rmb * Rb) / rm;
			double Tm = VOld[bi](i,j,k,iT);
			double Cs = sqrt(gam * R * Tm);
			VNew[bi](i,j,k,iCs) = Cs;
			if (Cs > maxCs) {
				maxCs = Cs;
			}
		});
	}
	
	if (isMMCs == 1) {//maxmaxCs
		amrex::ParallelFor(S_old, [=] AMREX_GPU_DEVICE (int bi, int i, int j, int k)
		{
			VNew[bi](i,j,k,iCs) = maxCs;
		});
	}
	
	double h = 1.0 / 3.0 * (hx_i + hy_j + hz_k);
	//++dt = beta * h / mmCsui_global;//% [2new].(75) //in main const
	/*+{//Check dt<=dt_test
		int i_u = 1;
		double mmCsui_global = 0.0;
		//for (k = 0; k < nc1_z; k++) { for (j = 0; j < nc1_y; j++) { for (i = 0; i < nc1_x; i++) {
		amrex::ParallelFor(S_old, [=, &mmCsui_global] AMREX_GPU_DEVICE (int bi, int i, int j, int k)
		{
			double ux = VOld[bi](i,j,k,iux), uy = VOld[bi](i,j,k,iuy), uz = VOld[bi](i,j,k,iuz);
			
			double ui = sqrt(ux * ux + uy * uy + uz * uz);
			double mmCsui_c = VNew[bi](i,j,k,iCs) + i_u * amrex::Math::abs(ui);
					if (mmCsui_c > mmCsui_global) {
						mmCsui_global = mmCsui_c;
					}
		});
		//}}}
		double beta = 0.1;
		double dt_test = beta * h / mmCsui_global;
		if (dt > dt_test)
		{
			std::cout << "QGD_advance.cpp dt_test = "<< dt_test << " vs dt = "<< dt << " time = " << time << "\n";
			dt = dt_test;
		}
	}+*/
	
    amrex::ParallelFor(S_old, [=] AMREX_GPU_DEVICE (int bi, int i, int j, int k)
    {
		double xira, yira, zira;//for test5

		double rm, rma, rmb;
		double cvm, cpm, gam;
		double sigma_a, sigma_b;
		
		double hx4, hy4, hz4;
		double hxy, hxz, hyz;

		double pm, Em, E_inm;
		double cs, Tm;
		double uxm, uym, uzm;
		double tau, visc, cond;
		
		double dpx, dpy, dpz, dTx, dTy, dTz;
		double duxx, duxy, duxz, duyx, duyy, duyz, duzx, duzy, duzz;
		double hWx1, hWy1, hWz1;
		double hwx, hwy, hwz;
		double droauxx, droauyy, droauzz, drobuxx, drobuyy, drobuzz;
		double Wxa2, Wxb2, Wya2, Wyb2, Wza2, Wzb2;
		double Froax, Frobx, Frox, Froay, Froby, Froy, Froaz, Frobz, Froz;
		double divu, Ptau;
		double PNSxx, PNSxy, PNSxz, PNSyx, PNSyy, PNSyz, PNSzx, PNSzy, PNSzz;
		double Pxx, Pxy, Pxz, Pyx, Pyy, Pyz, Pzx, Pzy, Pzz;
		double Fuxx, Fuxy, Fuxz, Fuyx, Fuyy, Fuyz, Fuzx, Fuzy, Fuzz;
		double E_in0m, droex, droey, droez, drox, droy, droz, qstx, qsty, qstz;
		double FEx, FEy, FEz;
		
		double b, c, d;
		double dts;
		double h;
		double adE_in;
		
		//if (i > nil && i < nc_x && j > nil && j < nc_y && k > nil && k < nc_z){
		/*--before me: already 0=VNew
        VNew[bi](i,j,k,ir) = 0.0; VNew[bi](i,j,k,ira) = 0.0; VNew[bi](i,j,k,irb) = 0.0;
		VNew[bi](i,j,k,iux) = 0.0; VNew[bi](i,j,k,iuy) = 0.0; VNew[bi](i,j,k,iuz) = 0.0;
		VNew[bi](i,j,k,iE) = 0.0;*/
		//test5
        /*VNew[bi](i,j,k,ir) = 2.*pow(10, -8); VNew[bi](i,j,k,ira) = pow(10, -8); VNew[bi](i,j,k,irb) = pow(10, -8);
		VNew[bi](i,j,k,iux) = pow(10, -8); VNew[bi](i,j,k,iuy) = pow(10, -8); VNew[bi](i,j,k,iuz) = pow(10, -8);
		VNew[bi](i,j,k,iE) = pow(10, -8);
		*/
		/*VNew[bi](i,j,k,iT) = VOld[bi](i,j,k,iT);
		VNew[bi](i,j,k,iVa) = VOld[bi](i,j,k,iVa);
		VNew[bi](i,j,k,iVb) = VOld[bi](i,j,k,iVb);
		VNew[bi](i,j,k,iE_in) = VOld[bi](i,j,k,iE_in);//pow(10, -8);
		*/
		//}
		
		//%%  X fluxes и Y fluxes
		//if (k > nil && k < nc_z)
		{
			hz4 = 4*dx[2];//hz[k + 1] + 2.0 * hz_k + hz[k - 1];
			//%%  X fluxes
			//if(1==0)
			//if (j > nil && j < nc_y && i < nc_x)
			{
				hy4 = 4*dx[1];//hy[j + 1] + 2.0 * hy_j + hy[j - 1]; //% 4 * hy_j;
			
				rm = (VOld[bi](i,j,k,ir) + VOld[bi](i+1,j,k,ir)) * 0.5;
				// upwind
				rma = (VOld[bi](i,j,k,ira) + VOld[bi](i+1,j,k,ira)) * 0.5; rmb = (VOld[bi](i,j,k,irb) + VOld[bi](i+1,j,k,irb)) * 0.5;
				// end upwind
				uxm = (VOld[bi](i,j,k,iux) + VOld[bi](i+1,j,k,iux)) * 0.5; uym = (VOld[bi](i,j,k,iuy) + VOld[bi](i+1,j,k,iuy)) * 0.5; uzm = (VOld[bi](i,j,k,iuz) + VOld[bi](i+1,j,k,iuz)) * 0.5;
				pm = (VOld[bi](i,j,k,ip) + VOld[bi](i+1,j,k,ip)) * 0.5;
				Em = (VOld[bi](i,j,k,iE) + VOld[bi](i+1,j,k,iE)) * 0.5;

				cvm = (rma * cva + rmb * cvb) / rm; cpm = (rma * cpa + rmb * cpb) / rm;
				gam = cpm / cvm;

				//++
				cs = (VNew[bi](i,j,k,iCs) + VNew[bi](i+1,j,k,iCs)) * 0.5;
				//-cs = (VOld[bi](i,j,k,iCs) + VOld[bi](i+1,j,k,iCs)) * 0.5;
				//-cs = VNew[bi](i,j,k,iCs);//test5
				/*if(k==0) //int(time/dt) == 1
				amrex::Print() << ", i= " << i << " j= " << j << " k= " << k
						<< " vk-1 = "<< VNew[bi](i+1,j,k,iCs) << " vk0 = "<< VNew[bi](i,j,k,iCs) << " \n";
				*/
				Tm = (VOld[bi](i,j,k,iT) + VOld[bi](i+1,j,k,iT)) * 0.5;

				//h = sqrt(hx_i * hy_j);
				h = sqrt(hx_i * hz_k);
				tau = alpha * h / (cs + i_t * amrex::Math::abs(pow(uxm, 2) + pow(uym, 2) + pow(uzm, 2)));
				//tau= 1e-11;//test5
				visc = tau * pm * Sc;// nu[2](78) or [1].(20)
				cond = tau * cpm * pm / Pr;// Злотник(78) (у него обратный)
				//cond = visc/(Pr*(gam-1));

				dpx = (VOld[bi](i+1,j,k,ip) - VOld[bi](i,j,k,ip)) / hx_i;
				dpy = (VOld[bi](i+1,j+1,k,ip) + VOld[bi](i,j+1,k,ip) - VOld[bi](i+1,j-1,k,ip) - VOld[bi](i,j-1,k,ip)) / hy4;
				dpz = (VOld[bi](i+1,j,k+1,ip) + VOld[bi](i,j,k+1,ip) - VOld[bi](i+1,j,k-1,ip) - VOld[bi](i,j,k-1,ip)) / hz4;
				
				//std::cout << "QGD_advance.cpp VOld[bi](i+1,j-1,k,ip) = "<< VOld[bi](i+1,j-1,k,ip) << " [i="<< i << ",j=" << j << ",k=" << k << "]\n";
				
				duxx = (VOld[bi](i+1,j,k,iux) - VOld[bi](i,j,k,iux)) / hx_i;
				duxy = (VOld[bi](i+1,j+1,k,iux) + VOld[bi](i,j+1,k,iux) - VOld[bi](i+1,j-1,k,iux) - VOld[bi](i,j-1,k,iux)) / hy4;
				duxz = (VOld[bi](i+1,j,k+1,iux) + VOld[bi](i,j,k+1,iux) - VOld[bi](i+1,j,k-1,iux) - VOld[bi](i,j,k-1,iux)) / hz4;

				duyx = (VOld[bi](i+1,j,k,iuy) - VOld[bi](i,j,k,iuy)) / hx_i;
				duyy = (VOld[bi](i+1,j+1,k,iuy) + VOld[bi](i,j+1,k,iuy) - VOld[bi](i+1,j-1,k,iuy) - VOld[bi](i,j-1,k,iuy)) / hy4;
				duyz = (VOld[bi](i+1,j,k+1,iuy) + VOld[bi](i,j,k+1,iuy) - VOld[bi](i+1,j,k-1,iuy) - VOld[bi](i,j,k-1,iuy)) / hz4;

				duzx = (VOld[bi](i+1,j,k,iuz) - VOld[bi](i,j,k,iuz)) / hx_i;
				duzy = (VOld[bi](i+1,j+1,k,iuz) + VOld[bi](i,j+1,k,iuz) - VOld[bi](i+1,j-1,k,iuz) - VOld[bi](i,j-1,k,iuz)) / hy4;
				duzz = (VOld[bi](i+1,j,k+1,iuz) + VOld[bi](i,j,k+1,iuz) - VOld[bi](i+1,j,k-1,iuz) - VOld[bi](i,j,k-1,iuz)) / hz4;

				hWx1 = tau * (rm * (uxm * duxx + uym * duxy + uzm * duxz) + dpx - rm * Fx);
				hWy1 = tau * (rm * (uxm * duyx + uym * duyy + uzm * duyz) + dpy - rm * Fy);
				hWz1 = tau * (rm * (uxm * duzx + uym * duzy + uzm * duzz) + dpz - rm * Fz);

				hwx = hWx1 / rm;

				droauxx = (VOld[bi](i+1,j,k,ira) * VOld[bi](i+1,j,k,iux) - VOld[bi](i,j,k,ira) * VOld[bi](i,j,k,iux)) / hx_i;
				droauyy = (VOld[bi](i+1,j+1,k,ira) * VOld[bi](i+1,j+1,k,iuy) + VOld[bi](i,j+1,k,ira) * VOld[bi](i,j+1,k,iuy) - VOld[bi](i+1,j-1,k,ira) * VOld[bi](i+1,j-1,k,iuy) - VOld[bi](i,j-1,k,ira) * VOld[bi](i,j-1,k,iuy)) / hy4;
				droauzz = (VOld[bi](i+1,j,k+1,ira) * VOld[bi](i+1,j,k+1,iuz) + VOld[bi](i,j,k+1,ira) * VOld[bi](i,j,k+1,iuz) - VOld[bi](i+1,j,k-1,ira) * VOld[bi](i+1,j,k-1,iuz) - VOld[bi](i,j,k-1,ira) * VOld[bi](i,j,k-1,iuz)) / hz4;

				drobuxx = (VOld[bi](i+1,j,k,irb) * VOld[bi](i+1,j,k,iux) - VOld[bi](i,j,k,irb) * VOld[bi](i,j,k,iux)) / hx_i;
				drobuyy = (VOld[bi](i+1,j+1,k,irb) * VOld[bi](i+1,j+1,k,iuy) + VOld[bi](i,j+1,k,irb) * VOld[bi](i,j+1,k,iuy) - VOld[bi](i+1,j-1,k,irb) * VOld[bi](i+1,j-1,k,iuy) - VOld[bi](i,j-1,k,irb) * VOld[bi](i,j-1,k,iuy)) / hy4;
				drobuzz = (VOld[bi](i+1,j,k+1,irb) * VOld[bi](i+1,j,k+1,iuz) + VOld[bi](i,j,k+1,irb) * VOld[bi](i,j,k+1,iuz) - VOld[bi](i+1,j,k-1,irb) * VOld[bi](i+1,j,k-1,iuz) - VOld[bi](i,j,k-1,irb) * VOld[bi](i,j,k-1,iuz)) / hz4;

				Wxa2 = tau * (droauxx + droauyy + droauzz) * uxm + rma * hwx;
				Wxb2 = tau * (drobuxx + drobuyy + drobuzz) * uxm + rmb * hwx;

				Froax = rma * uxm - Wxa2; Frobx = rmb * uxm - Wxb2; Frox = Froax + Frobx;

				divu = duxx + duyy + duzz;
				Ptau = tau * ((uxm * dpx + uym * dpy + uzm * dpz) + rm * pow(cs, 2) * divu - pow(cs, 2) / (gam * cvm * Tm) * Q);//[new2].(51)

				PNSxx = 2.0 * visc * duxx - 2.0 / 3.0 * visc * divu;
				PNSyx = visc * (duyx + duxy) + 0.0; 
				PNSzx = visc * (duzx + duxz) + 0.0;

				Pxx = PNSxx + uxm * hWx1 + Ptau;
				Pxy = PNSyx + uxm * hWy1;
				Pxz = PNSzx + uxm * hWz1;

				Fuxx = pm + uxm * Frox - Pxx;
				Fuyx = uym * Frox - Pxy;
				Fuzx = uzm * Frox - Pxz;

				dTx = (VOld[bi](i+1,j,k,iT) - VOld[bi](i,j,k,iT)) / hx_i;

				E_in0m = (VOld[bi](i,j,k,iE_in0) + VOld[bi](i+1,j,k,iE_in0)) * 0.5;
				droex = (VOld[bi](i+1,j,k,ir) * VOld[bi](i+1,j,k,iE_in) - VOld[bi](i,j,k,ir) * VOld[bi](i,j,k,iE_in)) / hx_i;
				droey = (VOld[bi](i+1,j+1,k,ir) * VOld[bi](i+1,j+1,k,iE_in) + VOld[bi](i,j+1,k,ir) * VOld[bi](i,j+1,k,iE_in) - VOld[bi](i+1,j-1,k,ir) * VOld[bi](i+1,j-1,k,iE_in) - VOld[bi](i,j-1,k,ir) * VOld[bi](i,j-1,k,iE_in)) / hy4;
				droez = (VOld[bi](i+1,j,k+1,ir) * VOld[bi](i+1,j,k+1,iE_in) + VOld[bi](i,j,k+1,ir) * VOld[bi](i,j,k+1,iE_in) - VOld[bi](i+1,j,k-1,ir) * VOld[bi](i+1,j,k-1,iE_in) - VOld[bi](i,j,k-1,ir) * VOld[bi](i,j,k-1,iE_in)) / hz4;

				drox = (VOld[bi](i+1,j,k,ir) - VOld[bi](i,j,k,ir)) / hx_i;
				droy = (VOld[bi](i+1,j+1,k,ir) + VOld[bi](i,j+1,k,ir) - VOld[bi](i+1,j-1,k,ir) - VOld[bi](i,j-1,k,ir)) / hy4;
				droz = (VOld[bi](i+1,j,k+1,ir) + VOld[bi](i,j,k+1,ir) - VOld[bi](i+1,j,k-1,ir) - VOld[bi](i,j,k-1,ir)) / hz4;

				qstx = tau * uxm * (uxm * (droex - (gam * cvm * Tm + E_in0m) * drox) +
					uym * (droey - (gam * cvm * Tm + E_in0m) * droy) +
					uzm * (droez - (gam * cvm * Tm + E_in0m) * droz) -
					Q);

				FEx = (Em + pm) * Frox / rm - cond * dTx - qstx - (Pxx * uxm + Pxy * uym + Pxz * uzm);

				hyz = hy_j * hz_k;

				VNew[bi](i,j,k,ira) = VNew[bi](i,j,k,ira) - Froax * hyz; 
				xira = VNew[bi](i,j,k,ira); 
				//yira = tau; zira = uxm;
				//yira = VOld[bi](i,j,k,iCs); zira = VNew[bi](i,j,k,iCs);
				//yira = tau; zira = cs;
				//yira = alpha * h / (cs); 
				//zira = alpha * h / (cs + i_t * amrex::Math::abs(pow(uxm, 2) + pow(uym, 2) + pow(uzm, 2)));
				//yira = Froax; zira = Wxa2;
				
				VNew[bi](i+1,j,k,ira) = VNew[bi](i+1,j,k,ira) + Froax * hyz;
				VNew[bi](i,j,k,irb) = VNew[bi](i,j,k,irb) - Frobx * hyz;
				VNew[bi](i+1,j,k,irb) = VNew[bi](i+1,j,k,irb) + Frobx * hyz;
				VNew[bi](i,j,k,iux) = VNew[bi](i,j,k,iux) - Fuxx * hyz;
				VNew[bi](i+1,j,k,iux) = VNew[bi](i+1,j,k,iux) + Fuxx * hyz;
				VNew[bi](i,j,k,iuy) = VNew[bi](i,j,k,iuy) - Fuyx * hyz;
				VNew[bi](i+1,j,k,iuy) = VNew[bi](i+1,j,k,iuy) + Fuyx * hyz;
				VNew[bi](i,j,k,iuz) = VNew[bi](i,j,k,iuz) - Fuzx * hyz;
				VNew[bi](i+1,j,k,iuz) = VNew[bi](i+1,j,k,iuz) + Fuzx * hyz;
				VNew[bi](i,j,k,iE) = VNew[bi](i,j,k,iE) - FEx * hyz;
				VNew[bi](i+1,j,k,iE) = VNew[bi](i+1,j,k,iE) + FEx * hyz;
			}
			//%%  Y fluxes
			//if(1==0)
			//if (i > nil && i < nc_x && j < nc_y)
			{
				hx4 = 4*dx[0];//hx[i + 1] + 2.0 * hx_i + hx[i - 1];
				
				rm = (VOld[bi](i,j,k,ir) + VOld[bi](i,j+1,k,ir)) * 0.5;
				// upwind
				rma = (VOld[bi](i,j,k,ira) + VOld[bi](i,j+1,k,ira)) * 0.5; rmb = (VOld[bi](i,j,k,irb) + VOld[bi](i,j+1,k,irb)) * 0.5;
				// end upwind
				uxm = (VOld[bi](i,j,k,iux) + VOld[bi](i,j+1,k,iux)) * 0.5; uym = (VOld[bi](i,j,k,iuy) + VOld[bi](i,j+1,k,iuy)) * 0.5; uzm = (VOld[bi](i,j,k,iuz) + VOld[bi](i,j+1,k,iuz)) * 0.5;
				pm = (VOld[bi](i,j,k,ip) + VOld[bi](i,j+1,k,ip)) * 0.5;
				Em = (VOld[bi](i,j,k,iE) + VOld[bi](i,j+1,k,iE)) * 0.5;

				cvm = (rma * cva + rmb * cvb) / rm; cpm = (rma * cpa + rmb * cpb) / rm;
				gam = cpm / cvm;

				//++
				cs = (VNew[bi](i,j,k,iCs) + VNew[bi](i,j+1,k,iCs)) * 0.5;
				//-cs = (VOld[bi](i,j,k,iCs) + VOld[bi](i,j+1,k,iCs)) * 0.5;
				//-cs = VNew[bi](i,j,k,iCs);//test5
				
				Tm = (VOld[bi](i,j,k,iT) + VOld[bi](i,j+1,k,iT)) * 0.5;

				h = sqrt(hx_i * hz_k);
				tau = alpha * h / (cs + i_t * amrex::Math::abs(pow(uxm, 2) + pow(uym, 2) + pow(uzm, 2)));
				//tau= 1e-11;//test5
				visc = tau * pm * Sc; // nu[2](78) or [1](20)
				cond = tau * cpm * pm / Pr; // Злотник(78)
				// cond = visc / (Pr * (gam - 1));

				dpx = (VOld[bi](i+1,j+1,k,ip) + VOld[bi](i+1,j,k,ip) - VOld[bi](i-1,j+1,k,ip) - VOld[bi](i-1,j,k,ip)) / hx4;
				dpy = (VOld[bi](i,j+1,k,ip) - VOld[bi](i,j,k,ip)) / hy_j;
				dpz = (VOld[bi](i,j+1,k+1,ip) + VOld[bi](i,j,k+1,ip) - VOld[bi](i,j+1,k-1,ip) - VOld[bi](i,j,k-1,ip)) / hz4;

				duxx = (VOld[bi](i+1,j+1,k,iux) + VOld[bi](i+1,j,k,iux) - VOld[bi](i-1,j+1,k,iux) - VOld[bi](i-1,j,k,iux)) / hx4;
				duxy = (VOld[bi](i,j+1,k,iux) - VOld[bi](i,j,k,iux)) / hy_j;
				duxz = (VOld[bi](i,j+1,k+1,iux) + VOld[bi](i,j,k+1,iux) - VOld[bi](i,j+1,k-1,iux) - VOld[bi](i,j,k-1,iux)) / hz4;

				duyx = (VOld[bi](i+1,j+1,k,iuy) + VOld[bi](i+1,j,k,iuy) - VOld[bi](i-1,j+1,k,iuy) - VOld[bi](i-1,j,k,iuy)) / hx4;
				duyy = (VOld[bi](i,j+1,k,iuy) - VOld[bi](i,j,k,iuy)) / hy_j;
				duyz = (VOld[bi](i,j+1,k+1,iuy) + VOld[bi](i,j,k+1,iuy) - VOld[bi](i,j+1,k-1,iuy) - VOld[bi](i,j,k-1,iuy)) / hz4;

				duzx = (VOld[bi](i+1,j+1,k,iuz) + VOld[bi](i+1,j,k,iuz) - VOld[bi](i-1,j+1,k,iuz) - VOld[bi](i-1,j,k,iuz)) / hx4;
				duzy = (VOld[bi](i,j+1,k,iuz) - VOld[bi](i,j,k,iuz)) / hy_j;
				duzz = (VOld[bi](i,j+1,k+1,iuz) + VOld[bi](i,j,k+1,iuz) - VOld[bi](i,j+1,k-1,iuz) - VOld[bi](i,j,k-1,iuz)) / hz4;

				hWx1 = tau * (rm * (uxm * duxx + uym * duxy + uzm * duxz) + dpx - rm * Fx);
				hWy1 = tau * (rm * (uxm * duyx + uym * duyy + uzm * duyz) + dpy - rm * Fy);
				hWz1 = tau * (rm * (uxm * duzx + uym * duzy + uzm * duzz) + dpz - rm * Fz);

				hwy = hWy1 / rm;

				droauxx = (VOld[bi](i+1,j+1,k,ira) * VOld[bi](i+1,j+1,k,iux) + VOld[bi](i+1,j,k,ira) * VOld[bi](i+1,j,k,iux) - VOld[bi](i-1,j+1,k,ira) * VOld[bi](i-1,j+1,k,iux) - VOld[bi](i-1,j,k,ira) * VOld[bi](i-1,j,k,iux)) / hx4;
				droauyy = (VOld[bi](i,j+1,k,ira) * VOld[bi](i,j+1,k,iuy) - VOld[bi](i,j,k,ira) * VOld[bi](i,j,k,iuy)) / hy_j;
				droauzz = (VOld[bi](i,j+1,k+1,ira) * VOld[bi](i,j+1,k+1,iuz) + VOld[bi](i,j,k+1,ira) * VOld[bi](i,j,k+1,iuz) - VOld[bi](i,j+1,k-1,ira) * VOld[bi](i,j+1,k-1,iuz) - VOld[bi](i,j,k-1,ira) * VOld[bi](i,j,k-1,iuz)) / hz4;

				drobuxx = (VOld[bi](i+1,j+1,k,irb) * VOld[bi](i+1,j+1,k,iux) + VOld[bi](i+1,j,k,irb) * VOld[bi](i+1,j,k,iux) - VOld[bi](i-1,j+1,k,irb) * VOld[bi](i-1,j+1,k,iux) - VOld[bi](i-1,j,k,irb) * VOld[bi](i-1,j,k,iux)) / hx4;
				drobuyy = (VOld[bi](i,j+1,k,irb) * VOld[bi](i,j+1,k,iuy) - VOld[bi](i,j,k,irb) * VOld[bi](i,j,k,iuy)) / hy_j;
				droauzz = (VOld[bi](i,j+1,k+1,irb) * VOld[bi](i,j+1,k+1,iuz) + VOld[bi](i,j,k+1,irb) * VOld[bi](i,j,k+1,iuz) - VOld[bi](i,j+1,k-1,irb) * VOld[bi](i,j+1,k-1,iuz) - VOld[bi](i,j,k-1,irb) * VOld[bi](i,j,k-1,iuz)) / hz4;

				Wya2 = tau * (droauxx + droauyy + droauzz) * uym + rma * hwy;
				Wyb2 = tau * (drobuxx + drobuyy + drobuzz) * uym + rmb * hwy;

				Froay = rma * uym - Wya2; Froby = rmb * uym - Wyb2; Froy = Froay + Froby;

				divu = duxx + duyy + duzz;
				Ptau = tau * ((uxm * dpx + uym * dpy + uzm * dpz) + rm * pow(cs, 2) * divu - pow(cs, 2) / (gam * cvm * Tm) * Q);

				PNSxy = visc * (duxy + duyx) + 0; 
				PNSyy = 2.0 * visc * duyy - 2.0 / 3.0 * visc * divu;  
				PNSzy = visc * (duzy + duyz) + 0;

				Pyx = PNSxy + uym * hWx1; 
				Pyy = PNSyy + uym * hWy1 + Ptau; 
				Pyz = PNSzy + uym * hWz1;

				Fuxy = uxm * Froy - Pyx; Fuyy = pm + uym * Froy - Pyy;  Fuzy = uzm * Froy - Pyz;

				dTy = (VOld[bi](i,j+1,k,iT) - VOld[bi](i,j,k,iT)) / hy_j;
				E_in0m = (VOld[bi](i,j,k,iE_in0) + VOld[bi](i,j+1,k,iE_in0)) * 0.5;
				droex = (VOld[bi](i+1,j+1,k,ir) * VOld[bi](i+1,j+1,k,iE_in) + VOld[bi](i+1,j,k,ir) * VOld[bi](i+1,j,k,iE_in) - VOld[bi](i-1,j+1,k,ir) * VOld[bi](i-1,j+1,k,iE_in) - VOld[bi](i-1,j,k,ir) * VOld[bi](i-1,j,k,iE_in)) / hx4;
				droey = (VOld[bi](i,j+1,k,ir) * VOld[bi](i,j+1,k,iE_in) - VOld[bi](i,j,k,ir) * VOld[bi](i,j,k,iE_in)) / hy_j;
				droez = (VOld[bi](i,j+1,k+1,ir) * VOld[bi](i,j+1,k+1,iE_in) + VOld[bi](i,j,k+1,ir) * VOld[bi](i,j,k+1,iE_in) - VOld[bi](i,j+1,k-1,ir) * VOld[bi](i,j+1,k-1,iE_in) - VOld[bi](i,j,k-1,ir) * VOld[bi](i,j,k-1,iE_in)) / hz4;

				drox = (VOld[bi](i+1,j+1,k,ir) + VOld[bi](i+1,j,k,ir) - VOld[bi](i-1,j+1,k,ir) - VOld[bi](i-1,j,k,ir)) / hx4;
				droy = (VOld[bi](i,j+1,k,ir) - VOld[bi](i,j,k,ir)) / hy_j;
				droz = (VOld[bi](i,j+1,k+1,ir) + VOld[bi](i,j,k+1,ir) - VOld[bi](i,j+1,k-1,ir) - VOld[bi](i,j,k-1,ir)) / hz4;

				qsty = tau * uym * (uxm * (droex - (gam * cvm * Tm + E_in0m) * drox) +
					uym * (droey - (gam * cvm * Tm + E_in0m) * droy) +
					uzm * (droez - (gam * cvm * Tm + E_in0m) * droz) -
					Q);

				FEy = (Em + pm) * Froy / rm - cond * dTy - qsty - (Pyx * uxm + Pyy * uym + Pyz * uzm);

				hxz = hx_i * hz_k;

				VNew[bi](i,j,k,ira) = VNew[bi](i,j,k,ira) - Froay * hxz; 
				//xira =   VOld[bi](i,j,k,iCs); 
				//yira = VNew[bi](i,j,k,iCs); zira = cs;//VNew[bi](i,j+1,k,iCs);
				//yira = VNew[bi](i,j,k,ira); 
				
				VNew[bi](i,j+1,k,ira) = VNew[bi](i,j+1,k,ira) + Froay * hxz;
				VNew[bi](i,j,k,irb) = VNew[bi](i,j,k,irb) - Froby * hxz;
				VNew[bi](i,j+1,k,irb) = VNew[bi](i,j+1,k,irb) + Froby * hxz;
				VNew[bi](i,j,k,iux) = VNew[bi](i,j,k,iux) - Fuxy * hxz;
				VNew[bi](i,j+1,k,iux) = VNew[bi](i,j+1,k,iux) + Fuxy * hxz;
				VNew[bi](i,j,k,iuy) = VNew[bi](i,j,k,iuy) - Fuyy * hxz;
				VNew[bi](i,j+1,k,iuy) = VNew[bi](i,j+1,k,iuy) + Fuyy * hxz;
				VNew[bi](i,j,k,iuz) = VNew[bi](i,j,k,iuz) - Fuzy * hxz;
				VNew[bi](i,j+1,k,iuz) = VNew[bi](i,j+1,k,iuz) + Fuzy * hxz;
				VNew[bi](i,j,k,iE) = VNew[bi](i,j,k,iE) - FEy * hxz;
				VNew[bi](i,j+1,k,iE) = VNew[bi](i,j+1,k,iE) + FEy * hxz;
			}
		}
		//%% Z fluxes
		//if(1==0)
		//if (i > nil && i < nc_x && j > nil && j < nc_y && k < nc_z)
		{
			hx4 = 4*dx[0];//hx[i + 1] + 2.0 * hx_i + hx[i - 1];
			hy4 = 4*dx[1];//hy[j + 1] + 2.0 * hy_j + hy[j - 1];
			
			rm = (VOld[bi](i,j,k,ir) + VOld[bi](i,j,k+1,ir)) * 0.5;
			// upwind
			rma = (VOld[bi](i,j,k,ira) + VOld[bi](i,j,k+1,ira)) * 0.5; rmb = (VOld[bi](i,j,k,irb) + VOld[bi](i,j,k+1,irb)) * 0.5;
			// end upwind
			uxm = (VOld[bi](i,j,k,iux) + VOld[bi](i,j,k+1,iux)) * 0.5; uym = (VOld[bi](i,j,k,iuy) + VOld[bi](i,j,k+1,iuy)) * 0.5; uzm = (VOld[bi](i,j,k,iuz) + VOld[bi](i,j,k+1,iuz)) * 0.5;
			pm = (VOld[bi](i,j,k,ip) + VOld[bi](i,j,k+1,ip)) * 0.5;
			Em = (VOld[bi](i,j,k,iE) + VOld[bi](i,j,k+1,iE)) * 0.5;

			cvm = (rma * cva + rmb * cvb) / rm; cpm = (rma * cpa + rmb * cpb) / rm;
			gam = cpm / cvm;

			//++
			cs = (VNew[bi](i,j,k,iCs) + VNew[bi](i,j,k+1,iCs)) * 0.5;
			//-cs = (VOld[bi](i,j,k,iCs) + VOld[bi](i,j,k+1,iCs)) * 0.5;
			//-cs = VNew[bi](i,j,k,iCs);//test5
					
			Tm = (VOld[bi](i,j,k,iT) + VOld[bi](i,j,k+1,iT)) * 0.5;

			h = sqrt(hx_i * hy_j);//??
			tau = alpha * h / (cs + i_t * amrex::Math::abs(pow(uxm, 2) + pow(uym, 2) + pow(uzm, 2)));
			//tau= 1e-11;//test5
			//tau = 1e-06;
			visc = tau * pm * Sc; // nu[2](78) or [1](20)
			cond = tau * cpm * pm / Pr; // Злотник(78)
			// cond = visc / (Pr * (gam - 1));
			//if(i==(nc1_x-1) && j==(nc1_y-1) && k==(nc1_z-1))
				//fprintf(stderr, "\n csz=%le csz1=%le %d,%d,%d", VOld[bi](i,j,k,iCs),VOld[bi](i,j,k+1,iCs), i,j,k);
				//fprintf(stderr, "\n csz=%le csz1=%le %d,%d,%d", VNew[bi](i,j,k,iCs),VNew[bi](i,j,k+1,iCs), i,j,k);
			//bnd Cs zero for VNew, but ok
			//amrex::Print() << ", i= " << i << " j= " << j << " k= " << k
			//				<< " v = "<< tau << " v2 = "<< h << " \n";
							
			dpx = (VOld[bi](i+1,j,k+1,ip) + VOld[bi](i+1,j,k,ip) - VOld[bi](i-1,j,k+1,ip) - VOld[bi](i-1,j,k,ip)) / hx4;
			dpx = (VOld[bi](i,j+1,k+1,ip) + VOld[bi](i,j+1,k,ip) - VOld[bi](i,j-1,k+1,ip) - VOld[bi](i,j-1,k,ip)) / hx4;
			dpz = (VOld[bi](i,j,k+1,ip) - VOld[bi](i,j,k,ip)) / hz_k;
			
			duxx = (VOld[bi](i+1,j,k+1,iux) + VOld[bi](i+1,j,k,iux) - VOld[bi](i-1,j,k+1,iux) - VOld[bi](i-1,j,k,iux)) / hx4;
			duxy = (VOld[bi](i,j+1,k+1,iux) + VOld[bi](i,j+1,k,iux) - VOld[bi](i,j-1,k+1,iux) - VOld[bi](i,j-1,k,iux)) / hy4;
			duxz = (VOld[bi](i,j,k+1,iux) - VOld[bi](i,j,k,iux)) / hz_k;

			duyx = (VOld[bi](i+1,j,k+1,iuy) + VOld[bi](i+1,j,k,iuy) - VOld[bi](i-1,j,k+1,iuy) - VOld[bi](i-1,j,k,iuy)) / hx4;
			duyy = (VOld[bi](i,j+1,k+1,iuy) + VOld[bi](i,j+1,k,iuy) - VOld[bi](i,j-1,k+1,iuy) - VOld[bi](i,j-1,k,iuy)) / hy4;
			duyz = (VOld[bi](i,j,k+1,iuy) - VOld[bi](i,j,k,iuy)) / hz_k;

			duzx = (VOld[bi](i+1,j,k+1,iuz) + VOld[bi](i+1,j,k,iuz) - VOld[bi](i-1,j,k+1,iuz) - VOld[bi](i-1,j,k,iuz)) / hx4;
			duzy = (VOld[bi](i,j+1,k+1,iuz) + VOld[bi](i,j+1,k,iuz) - VOld[bi](i,j-1,k+1,iuz) - VOld[bi](i,j-1,k,iuz)) / hy4;
			duzz = (VOld[bi](i,j,k+1,iuz) - VOld[bi](i,j,k,iuz)) / hz_k;

			hWx1 = tau * (rm * (uxm * duxx + uym * duxy + uzm * duxz) + dpx - rm * Fx);
			hWy1 = tau * (rm * (uxm * duyx + uym * duyy + uzm * duyz) + dpy - rm * Fy);
			hWz1 = tau * (rm * (uxm * duzx + uym * duzy + uzm * duzz) + dpz - rm * Fz);

			hwz = hWz1 / rm;

			droauxx = (VOld[bi](i+1,j,k+1,ira) * VOld[bi](i+1,j,k+1,iux) + VOld[bi](i+1,j,k,ira) * VOld[bi](i+1,j,k,iux) - VOld[bi](i-1,j,k+1,ira) * VOld[bi](i-1,j,k+1,iux) - VOld[bi](i-1,j,k,ira) * VOld[bi](i-1,j,k,iux)) / hx4;
			droauyy = (VOld[bi](i,j+1,k+1,ira) * VOld[bi](i,j+1,k+1,iuy) + VOld[bi](i,j+1,k,ira) * VOld[bi](i,j+1,k,iuy) - VOld[bi](i,j-1,k+1,ira) * VOld[bi](i,j-1,k+1,iuy) - VOld[bi](i,j-1,k,ira) * VOld[bi](i,j-1,k,iuy)) / hy4;
			droauzz = (VOld[bi](i,j,k+1,ira) * VOld[bi](i,j,k+1,iuz) - VOld[bi](i,j,k,ira) * VOld[bi](i,j,k,iuz)) / hz_k;

			drobuxx = (VOld[bi](i+1,j,k+1,irb) * VOld[bi](i+1,j,k+1,iux) + VOld[bi](i+1,j,k,irb) * VOld[bi](i+1,j,k,iux) - VOld[bi](i-1,j,k+1,irb) * VOld[bi](i-1,j,k+1,iux) - VOld[bi](i-1,j,k,irb) * VOld[bi](i-1,j,k,iux)) / hx4;
			droauyy = (VOld[bi](i,j+1,k+1,irb) * VOld[bi](i,j+1,k+1,iuz) + VOld[bi](i,j+1,k,irb) * VOld[bi](i,j+1,k,iuz) - VOld[bi](i,j-1,k+1,irb) * VOld[bi](i,j-1,k+1,iuz) - VOld[bi](i,j-1,k,irb) * VOld[bi](i,j-1,k,iuz)) / hy4;
			drobuzz = (VOld[bi](i,j,k+1,irb) * VOld[bi](i,j,k+1,iuz) - VOld[bi](i,j,k,irb) * VOld[bi](i,j,k,iuz)) / hz_k;

			Wza2 = tau * (droauxx + droauyy + droauzz) * uzm + rma * hwz;
			Wzb2 = tau * (drobuxx + drobuyy + drobuzz) * uzm + rmb * hwz;

			Froaz = rma * uzm - Wza2; Frobz = rmb * uzm - Wzb2; Froz = Froaz + Frobz;

			divu = duxx + duyy + duzz;
			Ptau = tau * ((uxm * dpx + uym * dpy + uzm * dpz) + rm * pow(cs, 2) * divu - pow(cs, 2) / (gam * cvm * Tm) * Q);

			PNSxz = visc * (duxz + duzx) + 0;
			PNSyz = visc * (duyz + duzy) + 0; 
			PNSzz = 2.0 * visc * duzz - 2.0 / 3.0 * visc * divu;
			Pzx = PNSxz + uzm * hWx1; 
			Pzy = PNSyz + uzm * hWy1; 
			Pzz = PNSzz + uzm * hWz1 + Ptau;
			
			Fuxz = uxm * Froz - Pzx; 
			Fuyz = uym * Froz - Pzy; 
			Fuzz = pm + uzm * Froz - Pzz;

			dTz = (VOld[bi](i,j,k+1,iT) - VOld[bi](i,j,k,iT)) / hz_k;
			E_in0m = (VOld[bi](i,j,k,iE_in0) + VOld[bi](i,j,k+1,iE_in0)) * 0.5;
			droex = (VOld[bi](i+1,j,k+1,ir) * VOld[bi](i+1,j,k+1,iE_in) + VOld[bi](i+1,j,k,ir) * VOld[bi](i+1,j,k,iE_in) - VOld[bi](i-1,j,k+1,ir) * VOld[bi](i-1,j,k+1,iE_in) - VOld[bi](i-1,j,k,ir) * VOld[bi](i-1,j,k,iE_in)) / hx4;
			droey = (VOld[bi](i,j+1,k+1,ir) * VOld[bi](i,j+1,k+1,iE_in) + VOld[bi](i,j+1,k,ir) * VOld[bi](i,j+1,k,iE_in) - VOld[bi](i,j-1,k+1,ir) * VOld[bi](i,j-1,k+1,iE_in) - VOld[bi](i,j-1,k,ir) * VOld[bi](i,j-1,k,iE_in)) / hy4;
			droez = (VOld[bi](i,j,k+1,ir) * VOld[bi](i,j,k+1,iE_in) - VOld[bi](i,j,k,ir) * VOld[bi](i,j,k,iE_in)) / hz_k;

			drox = (VOld[bi](i+1,j,k+1,ir) + VOld[bi](i+1,j,k,ir) - VOld[bi](i-1,j,k+1,ir) - VOld[bi](i-1,j,k,ir)) / hx4;
			droy = (VOld[bi](i,j+1,k+1,ir) + VOld[bi](i,j+1,k,ir) - VOld[bi](i,j-1,k+1,ir) - VOld[bi](i,j-1,k,ir)) / hy4;
			droz = (VOld[bi](i,j,k+1,ir) - VOld[bi](i,j,k,ir)) / hz_k;

			qstz = tau * uzm * (uxm * (droex - (gam * cvm * Tm + E_in0m) * drox) +
				uym * (droey - (gam * cvm * Tm + E_in0m) * droy) +
				uzm * (droez - (gam * cvm * Tm + E_in0m) * droz) -
				Q);

			FEz = (Em + pm) * Froz / rm - cond * dTz - qstz - (Pzx * uxm + Pzy * uym + Pzz * uzm);

			hxy = hx_i * hy_j;

			VNew[bi](i,j,k,ira) = VNew[bi](i,j,k,ira) - Froaz * hxy; 
			zira = VNew[bi](i,j,k,ira);
			xira = tau; yira = uzm; 
			//xira = alpha * h ; yira = cs; 
			
			
			//if((k+1)==4 && VNew[bi](i,j,k+1,iuz)!=0.0){
			//	amrex::Print() << ", vk+1 = "<< VNew[bi](i,j,k+1,iuz) << " \n"; all good
			//}
			//if((k)==0 && VNew[bi](i,j,k-1,iuz)!=0.0){
			//	amrex::Print() << ", i= " << i << " j= " << j << " k= " << k
			//		<< "vk-1 = "<< VNew[bi](i,j,k-1,iuz) << " \n";
			//}
			//amrex::Print() << ", k = "<< k << " \n";
		
			VNew[bi](i,j,k+1,ira) = VNew[bi](i,j,k+1,ira) + Froaz * hxy;
			VNew[bi](i,j,k,irb) = VNew[bi](i,j,k,irb) - Frobz * hxy;
			VNew[bi](i,j,k+1,irb) = VNew[bi](i,j,k+1,irb) + Frobz * hxy;
			VNew[bi](i,j,k,iux) = VNew[bi](i,j,k,iux) - Fuxz * hxy;
			VNew[bi](i,j,k+1,iux) = VNew[bi](i,j,k+1,iux) + Fuxz * hxy;
			VNew[bi](i,j,k,iuy) = VNew[bi](i,j,k,iuy) - Fuyz * hxy;
			VNew[bi](i,j,k+1,iuy) = VNew[bi](i,j,k+1,iuy) + Fuyz * hxy;
			VNew[bi](i,j,k,iuz) = VNew[bi](i,j,k,iuz) - Fuzz * hxy;
			VNew[bi](i,j,k+1,iuz) = VNew[bi](i,j,k+1,iuz) + Fuzz * hxy;
			VNew[bi](i,j,k,iE) = VNew[bi](i,j,k,iE) - FEz * hxy;
			VNew[bi](i,j,k+1,iE) = VNew[bi](i,j,k+1,iE) + FEz * hxy;
		}
		
		// New variables. Saved
		VNew[bi](i,j,k,iE_in0) = VOld[bi](i,j,k,iE_in0);
		// New variables
		dts = dt / (hx_i * hy_j * hz_k);
		h = 1.0/3. * (hx_i + hy_j + hz_k);
		//if (i > nil && i < nc_x && j > nil && j < nc_y && k > nil && k < nc_z)
		{
			xira = VOld[bi](i,j,k,ira); yira = VNew[bi](i,j,k,ira); zira = dts;
			VNew[bi](i,j,k,ira) = VNew[bi](i,j,k,ira) * dts + VOld[bi](i,j,k,ira); VNew[bi](i,j,k,irb) = VNew[bi](i,j,k,irb) * dts + VOld[bi](i,j,k,irb);
			
			//fprintf(stderr, "\nold in newira=%le xira=%le yira=%le zira=%le ", VNew[bi](i,j,k,ira), xira, yira,zira);
			if (VNew[bi](i,j,k,ira) < 0.0)
				VNew[bi](i,j,k,ira) = 1.0 * pow(10, -8.);
			if (VNew[bi](i,j,k,irb) < 0.0)
				VNew[bi](i,j,k,irb) = 1.0 * pow(10, -8.);

			VNew[bi](i,j,k,iux) = VNew[bi](i,j,k,iux) * dts + VOld[bi](i,j,k,ir) * VOld[bi](i,j,k,iux); // +dt.*RSux;
			VNew[bi](i,j,k,iuy) = VNew[bi](i,j,k,iuy) * dts + VOld[bi](i,j,k,ir) * VOld[bi](i,j,k,iuy); // +dt.*RSuy;
			VNew[bi](i,j,k,iuz) = VNew[bi](i,j,k,iuz) * dts + VOld[bi](i,j,k,ir) * VOld[bi](i,j,k,iuz); // +dt.*RSuz;
			VNew[bi](i,j,k,iE) = VNew[bi](i,j,k,iE) * dts + VOld[bi](i,j,k,iE); // +dt * RSE;

			//+VNew[bi](i,j,0,iuz) = 0;
			
			VNew[bi](i,j,k,ir) = VNew[bi](i,j,k,ira) + VNew[bi](i,j,k,irb);
			VNew[bi](i,j,k,iux) = VNew[bi](i,j,k,iux) / VNew[bi](i,j,k,ir); VNew[bi](i,j,k,iuy) = VNew[bi](i,j,k,iuy) / VNew[bi](i,j,k,ir); VNew[bi](i,j,k,iuz) = VNew[bi](i,j,k,iuz) / VNew[bi](i,j,k,ir);
			VNew[bi](i,j,k,iE_in) = (VNew[bi](i,j,k,iE) - 0.5 * VNew[bi](i,j,k,ir) * (VNew[bi](i,j,k,iux) * VNew[bi](i,j,k,iux) + VNew[bi](i,j,k,iuy) * VNew[bi](i,j,k,iuy) + VNew[bi](i,j,k,iuz) * VNew[bi](i,j,k,iuz))) / VNew[bi](i,j,k,ir);
			adE_in = VNew[bi](i,j,k,iE_in) - VNew[bi](i,j,k,iE_in0);

			cvm = (VNew[bi](i,j,k,ira) * cva + VNew[bi](i,j,k,irb) * cvb) / VNew[bi](i,j,k,ir);
			cpm = (VNew[bi](i,j,k,ira) * cpa + VNew[bi](i,j,k,irb) * cpb) / VNew[bi](i,j,k,ir);
			gam = cpm / cvm;

			sigma_a = Ra * VNew[bi](i,j,k,ira) / (cvm * VNew[bi](i,j,k,ir));//2.(18)
			sigma_b = Rb * VNew[bi](i,j,k,irb) / (cvm * VNew[bi](i,j,k,ir));
			b = sigma_a * (VNew[bi](i,j,k,ir) * adE_in - pa_inf) - pa_inf + sigma_b * (VNew[bi](i,j,k,ir) * adE_in - pb_inf) - pb_inf;
			c = (sigma_a * pb_inf + sigma_b * pa_inf) * VNew[bi](i,j,k,ir) * adE_in - gam * pa_inf * pb_inf;
			d = pow(b, 2.) + 4.0 * c;
			if (d <= 0) {
				fprintf(stderr, "\nError in d: i=%d j=%d k=%d, bi=%d, d=%g, step=%d, time=%.8le"
					, i, j, k, bi, d, int(time/dt), time);
				exit(EXIT_FAILURE);
			}
			VNew[bi](i,j,k,ip) = 0.5 * (b + sqrt(d));
			
			if ((VNew[bi](i,j,k,ip) < 0.0) || isnan(VNew[bi](i,j,k,ip))) {//еще проверку на complex number - Комплексное число
				fprintf(stderr, "\nError in p: i=%d j=%d k=%d, bi=%d, p=%g, "
					"step = %d, time = %.8le", 
					i, j, k, bi, VNew[bi](i,j,k,ip), 
					int(time/dt), time);//--ncycle, time);// --iteration, time);
				fprintf(stderr, "\nOther values: ro=%le, E=%le, E_in=%le, ux=%le, uy=%le, uz=%le\n"
					, VNew[bi](i,j,k,ir), VNew[bi](i,j,k,iE), VNew[bi](i,j,k,iE_in), VNew[bi](i,j,k,iux), VNew[bi](i,j,k,iuy), VNew[bi](i,j,k,iuz));
				double xbeg = -15*pow(10,-3);//A_3D
				fprintf(stderr, "\nxyz in (%le, %le, %le)", xbeg+i*hx_i, j*hy_j, k*hz_k);
				fprintf(stderr, "\nold in newira=%le xira=%le yira=%le zira=%le "
					, VNew[bi](i,j,k,ira), xira, yira,zira);
				exit(EXIT_FAILURE);
			}
			
			VNew[bi](i,j,k,iT) = Ra * VNew[bi](i,j,k,ira) / (VNew[bi](i,j,k,ip) + pa_inf) + Rb * VNew[bi](i,j,k,irb) / (VNew[bi](i,j,k,ip) + pb_inf);//[new2].(42)
			VNew[bi](i,j,k,iT) = 1.0 / VNew[bi](i,j,k,iT);
			VNew[bi](i,j,k,iVa) = Ra * VNew[bi](i,j,k,ira) * VNew[bi](i,j,k,iT) / (VNew[bi](i,j,k,ip) + pa_inf); //[2].(48) or [new2].(41)
			VNew[bi](i,j,k,iVb) = Rb * VNew[bi](i,j,k,irb) * VNew[bi](i,j,k,iT) / (VNew[bi](i,j,k,ip) + pb_inf);
			
			/*if(i==0 && j==0 && k==0 && int(time/dt) == 1)//0)
			{
				amrex::Print() << ", p: " << VOld[bi](i,j,k,ip) << " vs "; amrex::Print()  << VNew[bi](i,j,k,ip); amrex::Print()  << " (i=" << i << ") \n";
				amrex::Print() << ", E: " << VOld[bi](i,j,k,iE) << " vs " << VNew[bi](i,j,k,iE) << "\n";
				amrex::Print() << ", r: " << VOld[bi](i,j,k,ir) << " vs " << VNew[bi](i,j,k,ir) << "\n";
				amrex::Print() << ", b vs c: " << b << " vs " << c << "\n";
				amrex::Print() << ", sigma a vs b: " << sigma_a << " vs " << sigma_b << "\n";
				amrex::Print() << ", E_in: " << VOld[bi](i,j,k,iE_in) << " vs " << VNew[bi](i,j,k,iE_in) << "\n";
				double c_11 = (sigma_a * pb_inf + sigma_b * pa_inf);
				double c_12 = VNew[bi](i,j,k,ir) * adE_in;
				amrex::Print() << ", adE_in, c 1 vs 2: " << adE_in << ", " << c_11 << " vs " << c_12 << "\n";
				for (int ip=0; ip<16; ip++){
					amrex::Print() << ", arr: " << ip << " v=" << VOld[bi](i,j,k,ip) << "\n";
					amrex::Print() << ", +-i1: " << VOld[bi](i-1,j,k,ip) << " vs " << VOld[bi](i+1,j,k,ip) << " \n";
					amrex::Print() << ", +-j1: " << VOld[bi](i,j-1,k,ip) << " vs " << VOld[bi](i,j+1,k,ip) << " \n";
					amrex::Print() << ", +-k1: " << VOld[bi](i,j,k-1,ip) << " vs " << VOld[bi](i,j,k+1,ip) << " \n";
				}
				for (int ip=0; ip<16; ip++){
					amrex::Print() << ", OvsN: " << ip << " v: " << VOld[bi](i,j,k,ip) << " vs " << VNew[bi](i,j,k,ip) << "\n";
					amrex::Print() << ", diff: " << ip << " => " << VNew[bi](i,j,k,ip) - VOld[bi](i,j,k,ip) << "\n";
				}
				//amrex::Print() << ", Sc vs Pr: " << Sc << " vs " << Pr << "\n";
				amrex::Print() << "old in newira=" << VNew[bi](i,j,k,ira) << " xira=" << xira <<" yira="<<yira << " zira=" << zira << "\n";
				//c=> E_in or E - init error irb
				exit(EXIT_FAILURE);
			}//*/
		}
		
    });
	
	//exit(EXIT_FAILURE);
	/*
    Real maxval = S_new.max(0);
    Real minval = S_new.min(0);
    amrex::Print() << "min/max rho = " << minval << "/" << maxval;
    maxval = S_new.max(4);
    minval = S_new.min(4);
    amrex::Print() << "  min/max Sc number = " << minval << "/" << maxval << "\n";
	*/

    return dt;
}
