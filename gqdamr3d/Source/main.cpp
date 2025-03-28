#include <AMReX.H>
#include <AMReX_Amr.H>
#include <AMReX_ParmParse.H>

#include <chrono>

#include <filesystem> // For checking if restart directory exists

using namespace amrex;

amrex::LevelBld* getLevelBld ();

// Custom class inheriting from Amr to make restart public
class MyAmr : public Amr {
public:
    MyAmr (LevelBld* lb) : Amr(lb) {}
    void publicRestart(const std::string& restart_chkfile) {
        Amr::restart(restart_chkfile);  // Call the protected restart from within the class
    }
};

int main (int argc, char* argv[])
{
    auto start = std::chrono::high_resolution_clock::now();
    amrex::Initialize(argc,argv);

    int  max_step = -1;
    Real strt_time = 0.0;
    Real stop_time = -1.0;
	
    std::string restart_chkfile; // Variable to store the checkpoint directory
    {
        ParmParse pp;
        pp.query("max_step", max_step);
        pp.query("stop_time", stop_time);
		
        pp.query("restart_chkfile", restart_chkfile); // Read checkpoint directory from input
		//-restart_chkfile = "chk00600";
		//+restart_chkfile = "/home/khaitaliev/amrex-work/amrex/Tutorials/gqdamr3d/Exec/WaterColumnInteraction/chk00600";
		//+restart_chkfile = "init";
    }
    if (max_step < 0 && stop_time < 0.0 && restart_chkfile.empty()) {
        amrex::Abort("Exiting because neither max_step, stop_time nor restart_chkfile is specified.");
    }
	
	
    {
    //    auto amr = std::make_unique<AmrShallowWater>(getLevelBld());
        MyAmr amr(getLevelBld());//Amr amr(getLevelBld());

        if (!restart_chkfile.empty() && restart_chkfile != "init") {
			if (!std::filesystem::exists(restart_chkfile)) {
				amrex::Abort("Restart checkpoint directory not found: " + restart_chkfile);
			}
            // Restart from a checkpoint
            amrex::Print() << "Restarting from checkpoint: " << restart_chkfile << "\n";
            amr.publicRestart(restart_chkfile); //amr.restart(restart_chkfile);
        } else {
            // Initialize from scratch
            amr.init(strt_time, stop_time);
        }

        while ( amr.okToContinue() &&
                (amr.levelSteps(0) < max_step || max_step < 0) &&
                (amr.cumTime() < stop_time || stop_time < 0.0) )
        {
            amr.coarseTimeStep(stop_time);
        }

        // Write final checkpoint and plotfile
        if (amr.stepOfLastCheckPoint() < amr.levelSteps(0)) {
            amr.checkPoint();
        }
        if (amr.stepOfLastPlotFile() < amr.levelSteps(0)) {
            amr.writePlotFile();
        }
    }

    amrex::Finalize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    std::cout << duration.count() << " seconds\n";
}
