#include "AmrQGD.H"
#include <cmath>

using namespace amrex;

void
AmrQGD::initData ()
{
	const auto problo = Geom().ProbLoArray();
	const auto dx = Geom().CellSizeArray();
	MultiFab& S_new = get_new_data(State_Type);
	auto const& snew = S_new.arrays();

	amrex::ParallelFor(S_new,
		[=] AMREX_GPU_DEVICE (int bi, int i, int j, int k) noexcept
	{
		Real x = problo[0] + (i+0.5)*dx[0];
		Real y = problo[1] + (j+0.5)*dx[1];

		//%%3.5 Shock/Water-Column interaction [3]Китамура
		double gma = gamma_a, gmb = gamma_b;
		
		double Ra = RGas_a;
		double Rb = RGas_b;
		double cva = Ra / (gma - 1); // 717.5; % Air
		double cvb = Rb / (gmb - 1); // 1495; % Water
        
		
		int ir = 0, ira = 1, irb = 2, 
		iux = 3, iuy = 4, iuz = 5,
		ip = 6, iE = 7, iE_in = 8, iE_in0 = 9,
		iT = 10, 
		iCs = 11,
		iVa = 12, iVb = 13;
		
		double pa_inf = painf;
		double pb_inf = pbinf;
		
			
		double xc = x - x_bub;//(x[i] + x[i - 1]) * 0.5 - x_bub;//??
		double xc0 = x;
		double yc = y;//(y[j] + y[j - 1]) * 0.5;
		//yc = y-0.005;//test

		if (x <= midX)
		{// Left
			snew[bi](i,j,k,ip) = pL;   						//p
			snew[bi](i,j,k,iux) = u1x;  						//Ux
			snew[bi](i,j,k,iuy) = u1y;  						//Uy
			snew[bi](i,j,k,iuz) = u1z;  						//Uz
			snew[bi](i,j,k,iT) = T1;    						//T
			snew[bi](i,j,k,iVa) = 1 - esp;					//Va
			snew[bi](i,j,k,iVb) = 1 - snew[bi](i,j,k,iVa);	//Vb
			snew[bi](i,j,k,iE_in0) = E_in01; 					//E_in0
        }
        else
		{// Right
			snew[bi](i,j,k,ip) = pR;    														//p
			snew[bi](i,j,k,iux) = u2x;  														//Ux
			snew[bi](i,j,k,iuy) = u2y;  														//Uy
			snew[bi](i,j,k,iuz) = u2z;  														//Uz
			snew[bi](i,j,k,iT) = T2;    														//T
			//snew[bi](i,j,k,iVa) = 1 - esp;  													//Va
			//snew[bi](i,j,k,iVb) = 1 - snew[bi](i,j,k,iVa);									//Vb
			snew[bi](i,j,k,iE_in0) = E_in02; 													//E_in0
			if (xc * xc + yc * yc >= r_bub * r_bub) {
				snew[bi](i,j,k,iVa) = 1 - esp;  snew[bi](i,j,k,iVb) = 1 - snew[bi](i,j,k,iVa);	//Va & Vb
			}
			else {
				snew[bi](i,j,k,iVa) = esp;  snew[bi](i,j,k,iVb) = 1 - snew[bi](i,j,k,iVa);		//Va & Vb
			}
        }
		double h = dx[0];//hx[i];
		double hx_i = dx[0];
		if (1 == 1 && (r_bub - 2 * hx_i) <= sqrt(xc0 * xc0 + yc * yc) && sqrt(xc0 * xc0 + yc * yc) <= (r_bub + 2 * h)) {// between
			double ksi2 = (sqrt(xc0 * xc0 + yc * yc) - (r_bub - 2 * h)) / (4 * h);
			double Gksi = -pow(ksi2, 2) * (2 * ksi2 - 3);
			snew[bi](i,j,k,iVb) = Gksi * esp + (1 - Gksi) * (1 - esp); snew[bi](i,j,k,iVa) = 1 - snew[bi](i,j,k,iVb);	//Va & Vb
		}
		snew[bi](i,j,k,ira) = snew[bi](i,j,k,iVa) * (snew[bi](i,j,k,ip) + pa_inf) / (Ra * snew[bi](i,j,k,iT));
		snew[bi](i,j,k,irb) = snew[bi](i,j,k,iVb) * (snew[bi](i,j,k,ip) + pb_inf) / (Rb * snew[bi](i,j,k,iT)); // [2].(8)
		snew[bi](i,j,k,ir) = snew[bi](i,j,k,ira) + snew[bi](i,j,k,irb);
		snew[bi](i,j,k,iE_in) = cva * snew[bi](i,j,k,iT) + snew[bi](i,j,k,iT) * Ra * pa_inf / (snew[bi](i,j,k,ip) + pa_inf) + snew[bi](i,j,k,iE_in0); // [2].(9) and wo [2].(4)
		snew[bi](i,j,k,iE_in) = snew[bi](i,j,k,ira) * snew[bi](i,j,k,iE_in) + snew[bi](i,j,k,irb) * (cvb * snew[bi](i,j,k,iT) + snew[bi](i,j,k,iT) * Rb * pb_inf / (snew[bi](i,j,k,ip) + pb_inf) + snew[bi](i,j,k,iE_in0));
		snew[bi](i,j,k,iE_in) = snew[bi](i,j,k,iE_in) / (snew[bi](i,j,k,ira) + snew[bi](i,j,k,irb));
		double ux = snew[bi](i,j,k,iux), uy = snew[bi](i,j,k,iuy), uz = snew[bi](i,j,k,iuz); 
		snew[bi](i,j,k,iE) = snew[bi](i,j,k,iE_in) * (snew[bi](i,j,k,ira) + snew[bi](i,j,k,irb)) + 0.5 * (snew[bi](i,j,k,ira) + snew[bi](i,j,k,irb)) * (ux * ux + uy * uy + uz * uz);
		/**/
		
		//not nedeed
		//snew[bi](i,j,k,iCs) = 0.;
		
		double cpa = cva*gma;//Air 1004.5
		double cpb = cvb*gmb;//Kitamura: Water 4186
		
		if (typeCs == 1) {
			// typeCs = 1 - according to Zlotnik's seminar, it is more correct
			// [new2].(33)
			double testCs1, testCs2;
			
			double ro = snew[bi](i,j,k,ir);
			double roa = snew[bi](i,j,k,ira); double rob = snew[bi](i,j,k,irb);
			double p = snew[bi](i,j,k,ip);
		
			double adE_in = snew[bi](i,j,k,iE_in) - snew[bi](i,j,k,iE_in0);
			double rm = snew[bi](i,j,k,ir), rma = snew[bi](i,j,k,ira), rmb = snew[bi](i,j,k,irb); // it is rmaf
			double cvm = (rma * cva + rmb * cvb) / rm;
			double cpm = (rma * cpa + rmb * cpb) / rm;
			double gam = cpm / cvm;
			double sigma_a = Ra * roa / (cvm * ro);//2.(18)
			double sigma_b = Rb * rob / (cvm * ro);
			// p is p_+
			double E1 = sigma_a * (ro * adE_in - pa_inf) / pow(p + pa_inf, 2.);
			E1 = E1 + sigma_b * (ro * adE_in - pb_inf) / pow(p + pb_inf, 2.); 

			E1 = 1.0 / E1; 
			E1 = sqrt(gam / ro * E1); 
			snew[bi](i,j,k,iCs) = E1;
		}
		else
		{
			amrex::Print() << "\n typeCs = "<< typeCs << " is not suitable for this task. Use typeCs = 1\n";
			exit(EXIT_FAILURE);//noexcept?
		}
    });
    FillPatcherFill(S_new, 0, ncomp, nghost, 0, State_Type, 0);
    amrex::Print() << "Amr QGD solver will start with next params: " << "AlphaQQD = " << alphaQgd << " and ScQGD = " << ScQgd << "\n\n";
}

