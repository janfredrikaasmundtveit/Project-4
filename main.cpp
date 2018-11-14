#include <cstdlib>
#include <iostream>
#include <cmath>
#include <iomanip>
#include  "mpi.h"
#include <ctime>
#include <fstream>
#include <string>
#include <time.h>
#include <random>
#include<armadillo>
#include "vectormatrixclass.h"

using namespace arma;
using namespace std;
ofstream ofile;


void output(double, int, int , double[],int);
void mcsampling(int, double, double,int,double[], mat &,double &,double &);


int main (int argc, char* argv[])
{ string filename;
  int NSpin, MCC, mc;
  double ITemp, FTemp, TempStep;
  int NProcesses, RankProcess;



//   MPI initializations
MPI_Init (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD, &NProcesses);
  MPI_Comm_rank (MPI_COMM_WORLD, &RankProcess);
// set up in master op

 if (RankProcess == 0 && argc <= 5) {
    cout << "Bad Usage: " << argv[0] << 
      " read output file, Number of spins, MC cycles, initial and final temperature and tempurate step" << endl;
    exit(1);
  } 
  if ((RankProcess == 0) && (argc > 1)) {
    filename=argv[1];
    NSpin = atoi(argv[2]);
    mc = atoi(argv[3]);  //mccycles  
    MCC=pow(10,mc);
    ITemp = atof(argv[4]);
    FTemp = atof(argv[5]);
    TempStep = atof(argv[6]);
  }



  if (RankProcess == 0) {
    string fileout = filename;
    string argument = to_string(NSpin);
    fileout.append(argument);
    ofile.open(fileout);
  }
 
  MPI_Bcast (&MCC, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast (&NSpin, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast (&ITemp, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast (&FTemp, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast (&TempStep, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double  TimeStart, TimeEnd, TotalTime;
  //TimeStart = MPI_Wtime();
 // setting groundstate=0
  mat lattice=ones<mat>(NSpin,NSpin); 
  double MagneticMoment=NSpin*NSpin; double Energy= NSpin*NSpin*(-2); 	 
  for (double temp=ITemp; temp < FTemp; temp+=TempStep)
  { 
  
  		
  	double LocE[]={0.0,0.0,0.0,0.0,0.0}; //E,E^2,M,M^2,|M|
		double TotE[]={0.0,0.0,0.0,0.0,0.0}; 		  
		//mcsampling
	 double pross= (double) RankProcess/((double)NProcesses);
	  mcsampling(MCC,temp,pross,NSpin,LocE,lattice,Energy,MagneticMoment);
  	  MPI_Reduce(&LocE, &TotE, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  	  
  	if ( RankProcess == 0){ 
  		output(temp,NSpin,MCC,TotE,NProcesses);	
      //cout << (TotE[1]-TotE[0]*TotE[0])/MCC << "\n";
		}

 }

    if(RankProcess == 0){ 
     ofile.close(); } // close output file
  
  //  End MPI
MPI_Finalize ();

return 0;
}

void mcsampling(int MCC, double temp, double pross,int NSpin,double LocE[], mat &lattice, double &Energy,double &MagneticMoment){
std::random_device rd;
  std::mt19937_64 gen(rd())
  // Set up the uniform distribution for x \in [[0, 1]
  std::uniform_real_distribution<double> RandomNumberGenerator(0.0,1.0);
    temp=temp+pross;
  	int totspin=NSpin*NSpin;
  	vec EnergyDifference=zeros<vec>(17); 
    //double* E= new double[4*totspin];

  for( int de =-8; de <= 8; de+=4){ EnergyDifference(de+8) = exp(-de/temp);} // using actual energy not just above groundtate

  		for (int i = 0; i < MCC; i++)
  			{
  			 for(int j=0; j<totspin; j++){
  			  	int ix = (int) (RandomNumberGenerator(gen)*(NSpin));
     			int iy = (int) (RandomNumberGenerator(gen)*(NSpin));
			
				
       
			int deltaE=2*lattice(ix,iy)*(lattice(ix,(iy+1)%NSpin)+lattice(ix,(iy+(NSpin)-1)%NSpin)+lattice((ix+1)%NSpin,iy)+lattice((ix+(NSpin)-1)%NSpin,iy));
				  
           
				  
   				if( RandomNumberGenerator(gen) <= EnergyDifference(deltaE+8)){
   					lattice(ix,iy) *=-1.0;
   				
   				MagneticMoment += 2.0*lattice(ix,iy);
			  	Energy = Energy+((double)(deltaE));
      
		
   				}	
   				
   		 }
       
       //ofile << setw(15) << setprecision(8) << i;
       //ile << setw(15) << setprecision(8) << Energy/totspin << "\n";
     
   		if(i>1000){
   		//int k= (int) Energy+NSpin*NSpin*(2);
      // E[k] +=1;
   		  LocE[0] +=Energy; LocE[1] +=Energy*Energy; LocE[2] +=MagneticMoment; LocE[3] +=MagneticMoment*MagneticMoment; LocE[4] +=fabs(MagneticMoment);
			    	//ofile << setw(15) << setprecision(8) << i;

   			//ofile << setw(15) << setprecision(8) << Energy << "\n";
			   }
    }
 
//   for (int l = 0; l <=4*totspin ; ++l)
  // { ofile << setw(15) << setprecision(8) << l-2*totspin;
    // ofile << setw(15) << setprecision(8) << E[l] << "\n";
  // }
}

void output(double temp, int NSpin, int MCC,double TotE[],int NProcesses){
//lattice size, cycles, temprature, other
 double norm = 1.0/((double) (MCC)*(double)(NProcesses));  // divided by  number of cycles
//groundstate 
 double n=1.0/((double)(MCC)); 
  double E_ExpectationValues = TotE[0]*norm; 
  double E2_ExpectationValues = TotE[1]*norm;	
  double M_ExpectationValues = TotE[2]*norm;
  double M2_ExpectationValues = TotE[3]*norm;
  double Mabs_ExpectationValues = TotE[4]*norm;  
  double AllSpins = 1.0/((double) NSpin*NSpin);
  double HeatCapacity = (E2_ExpectationValues- E_ExpectationValues*E_ExpectationValues)*AllSpins/(temp*temp);
  double MagneticSusceptibility = (M2_ExpectationValues - Mabs_ExpectationValues*Mabs_ExpectationValues)*AllSpins/(temp);
  ofile << setiosflags(ios::showpoint | ios::uppercase);
  ofile << setw(15) << setprecision(8) << temp;
  ofile << setw(15) << setprecision(8) << (E_ExpectationValues)*AllSpins;
  ofile << setw(15) << setprecision(8) << HeatCapacity;
  ofile << setw(15) << setprecision(8) << M_ExpectationValues*AllSpins;
  ofile << setw(15) << setprecision(8) << MagneticSusceptibility;
  ofile << setw(15) << setprecision(8) << Mabs_ExpectationValues*AllSpins << "\n";



}