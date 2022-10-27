#include "TString.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TStopwatch.h"
#include "TF1.h"
#include "TMath.h"

//---------------------------------------------------------------------------------------------------------//
// Find Distance between Muon Vertex and Photon Production Point (lambda)
//---------------------------------------------------------------------------------------------------------//
double find_lambda(double xmu_rec,double ymu_rec,double zmu_rec,double xrecDir,double yrecDir,double zrecDir,double x_pmtpos,double y_pmtpos,double z_pmtpos,double theta_cher)
{
     double lambda1 = 0.0;    double lambda2 = 0.0;    double length = 0.0 ;
     double xmupos_t1 = 0.0;  double ymupos_t1 = 0.0;  double zmupos_t1 = 0.0;
     double xmupos_t2 = 0.0;  double ymupos_t2 = 0.0;  double zmupos_t2 = 0.0;
     double xmupos_t = 0.0;   double ymupos_t = 0.0;   double zmupos_t = 0.0;

     double theta_muDir_track = 0.0;
     double theta_muDir_track1 = 0.0;  double theta_muDir_track2 = 0.0;
     double cos_thetacher = cos(theta_cher*TMath::DegToRad());
     double xmupos_tmin = 0.0; double ymupos_tmin = 0.0; double zmupos_tmin = 0.0;
     double xmupos_tmax = 0.0; double ymupos_tmax = 0.0; double zmupos_tmax = 0.0;
     double lambda_min = 10000000;  double lambda_max = -99999999.9;  double lambda = 0.0;

     double alpha = (xrecDir*xrecDir + yrecDir*yrecDir + zrecDir*zrecDir) * ( (xrecDir*xrecDir + yrecDir*yrecDir + zrecDir*zrecDir) - (cos_thetacher*cos_thetacher) );
     double beta = ( (-2)*(xrecDir*(x_pmtpos - xmu_rec) + yrecDir*(y_pmtpos - ymu_rec) + zrecDir*(z_pmtpos - zmu_rec) )*((xrecDir*xrecDir + yrecDir*yrecDir + zrecDir*zrecDir) - (cos_thetacher*cos_thetacher)) );
     double gamma = ( ( (xrecDir*(x_pmtpos - xmu_rec) + yrecDir*(y_pmtpos - ymu_rec) + zrecDir*(z_pmtpos - zmu_rec))*(xrecDir*(x_pmtpos - xmu_rec) + yrecDir*(y_pmtpos - ymu_rec) + zrecDir*(z_pmtpos - zmu_rec)) ) - (((x_pmtpos - xmu_rec)*(x_pmtpos - xmu_rec) + (y_pmtpos - ymu_rec)*(y_pmtpos - ymu_rec) + (z_pmtpos - zmu_rec)*(z_pmtpos - zmu_rec))*(cos_thetacher*cos_thetacher)) );


     double discriminant = ( (beta*beta) - (4*alpha*gamma) );

     lambda1 = ( (-beta + sqrt(discriminant))/(2*alpha));
     lambda2 = ( (-beta - sqrt(discriminant))/(2*alpha));

     xmupos_t1 = xmu_rec + xrecDir*lambda1;  xmupos_t2 = xmu_rec + xrecDir*lambda2;
     ymupos_t1 = ymu_rec + yrecDir*lambda1;  ymupos_t2 = ymu_rec + yrecDir*lambda2;
     zmupos_t1 = zmu_rec + zrecDir*lambda1;  zmupos_t2 = zmu_rec + zrecDir*lambda2;

     double tr1 = sqrt((x_pmtpos - xmupos_t1)*(x_pmtpos - xmupos_t1) + (y_pmtpos - ymupos_t1)*(y_pmtpos - ymupos_t1) + (z_pmtpos - zmupos_t1)*(z_pmtpos - zmupos_t1));
     double tr2 = sqrt((x_pmtpos - xmupos_t2)*(x_pmtpos - xmupos_t2) + (y_pmtpos - ymupos_t2)*(y_pmtpos - ymupos_t2) + (z_pmtpos - zmupos_t2)*(z_pmtpos - zmupos_t2));
     theta_muDir_track1 = (acos( (xrecDir*(x_pmtpos - xmupos_t1) + yrecDir*(y_pmtpos - ymupos_t1) + zrecDir*(z_pmtpos - zmupos_t1))/(tr1) )*TMath::RadToDeg());
     theta_muDir_track2 = (acos( (xrecDir*(x_pmtpos - xmupos_t2) + yrecDir*(y_pmtpos - ymupos_t2) + zrecDir*(z_pmtpos - zmupos_t2))/(tr2) )*TMath::RadToDeg());
     //---------------------------- choose lambda!! ---------------------------------
     if( theta_muDir_track1 < theta_muDir_track2 ){
       lambda = lambda1;
       xmupos_t = xmupos_t1;
       ymupos_t = ymupos_t1;
       zmupos_t = zmupos_t1;
       theta_muDir_track=theta_muDir_track1;
     }else if( theta_muDir_track2 < theta_muDir_track1 ){
       lambda = lambda2;
       xmupos_t = xmupos_t2;
       ymupos_t = ymupos_t2;
       zmupos_t = zmupos_t2;
       theta_muDir_track=theta_muDir_track2;
     }

     return lambda;
}
     

void LocalFindTrackLengthInWater()
{

   int ievt=0; float trueNeuE=0; float trueE=0;
   float TrueTrackLengthInWater2=0; float TrueTrackLengthInMrd2=0; float recoDWallR2=0; float recoDWallZ2=0;
   float diffDirAbs2=0; 
   float dirX2=0; float dirY2=0; float dirZ2=0;
   float vtxX2=0; float vtxY2=0; float vtxZ2=0;

   ofstream csvfile;
   //csvfile.open ("data_forRecoLength_05202019.csv");
   //csvfile.open ("data/data_forRecoLength_06082019.csv");
   //csvfile.open ("data/data_forRecoLength_06082019CC0pi.csv"); //events with no pi's
   csvfile.open ("data_forRecoLength_beamlikeEvts.csv");

   int maxhits0=1100; 
      //--- write to file: ---//
      //if(first==1 && deny_access==0){
      //    deny_access=1;
          for (int i=0; i<maxhits0;++i){
             stringstream strs;
             strs << i;
             string temp_str = strs.str();
             string X_name= "l_";
             X_name.append(temp_str);
             const char * xname = X_name.c_str();
             csvfile<<xname<<",";
          }
          for (int ii=0; ii<maxhits0;++ii){
             stringstream strs4;
             strs4 << ii;
             string temp_str4 = strs4.str();
             string T_name= "T_";
             T_name.append(temp_str4);
             const char * tname = T_name.c_str();
             csvfile<<tname<<",";
          }
        csvfile<<"lambda_max"<<","; //first estimation of track length(using photons projection on track)
        csvfile<<"totalPMTs"<<",";
        csvfile<<"totalLAPPDs"<<",";
        csvfile<<"lambda_max"<<",";
        csvfile<<"TrueTrackLengthInWater"<<",";
        //csvfile<<"neutrinoE"<<",";    //comment
        csvfile<<"trueKE"<<",";
        csvfile<<"diffDirAbs"<<",";
        csvfile<<"TrueTrackLengthInMrd"<<",";
        csvfile<<"recoDWallR"<<",";
        csvfile<<"recoDWallZ"<<",";
        csvfile<<"dirX"<<",";
        csvfile<<"dirY"<<",";
        csvfile<<"dirZ"<<",";
        csvfile<<"vtxX"<<",";
        csvfile<<"vtxY"<<",";
        csvfile<<"vtxZ"<<",";
        csvfile<<"truedirX"<<",";
        csvfile<<"truedirY"<<",";
        csvfile<<"truedirZ"<<",";
        csvfile<<"truevtxX"<<",";
        csvfile<<"truevtxY"<<",";
        csvfile<<"truevtxZ"<<",";
        csvfile<<"recoVtxFOM";
        //csvfile<<"recoStatus"<<",";  //comment
        //csvfile<<"deltaVtxR"<<",";   //comment
        //csvfile<<"deltaAngle";       //comment
        csvfile<<'\n';


   char fname[100]; int count1=0, count2=0;
   //sprintf(fname,"PhaseIIReco_SuccessfulReco_04202019.root");//,i);
   //sprintf(fname,"PMTLAPPDReco_743Runs_05202019.root");//,i);
   //sprintf(fname,"data/PMTLAPPDReco_All_06082019.root");
   //sprintf(fname,"/home/liliadrak/ANNIE/ntuples_Ereco/vtxreco-beamlikegridall-cut.root");
   sprintf(fname,"vtxreco-beamlikemrd.root");
   TFile *input=new TFile(fname,"READONLY");
   cout<<"input file: "<<fname<<endl;
   
   TTree *regTree= (TTree*) input->Get("phaseIITriggerTree");
   //----------- read the tree from file:
   //deny_access=1;
   Int_t run, event, nhits, trigger, recoStatus;
   double recoVtxFOM, deltaVtxR, deltaAngle;
   double truevtxX,truevtxY,truevtxZ,truedirX,truedirY,truedirZ,vtxX,vtxY,vtxZ,dirX,dirY,dirZ,TrueTrackLengthInMrd,TrueTrackLengthInWater,TrueNeutrinoEnergy,trueEnergy,TrueMomentumTransfer,TrueMuonAngle;
   std::string *TrueInteractionType = 0;
   std::vector<double> *digitX=0; std::vector<double> *digitY=0;  std::vector<double> *digitZ=0;
   std::vector<double> *digitT=0; std::vector<int>  *digitType=0;
   float trueMuonEnergy=0.;
   int Pi0Count,PiPlusCount,PiMinusCount;
   double DR;

   regTree->SetBranchAddress("runNumber", &run);
   regTree->SetBranchAddress("eventNumber", &event);
   regTree->SetBranchAddress("trueMuonEnergy", &trueEnergy);
   //regTree->SetBranchAddress("TrueNeutrinoEnergy", &TrueNeutrinoEnergy);
   //regTree->SetBranchAddress("trigger", &trigger);
   regTree->SetBranchAddress("nhits", &nhits);
   regTree->SetBranchAddress("recoVtxX", &vtxX);
   regTree->SetBranchAddress("recoVtxY", &vtxY);
   regTree->SetBranchAddress("recoVtxZ", &vtxZ);
   regTree->SetBranchAddress("recoDirX", &dirX);
   regTree->SetBranchAddress("recoDirY", &dirY);
   regTree->SetBranchAddress("recoDirZ", &dirZ);
   regTree->SetBranchAddress("trueVtxX", &truevtxX);
   regTree->SetBranchAddress("trueVtxY", &truevtxY);
   regTree->SetBranchAddress("trueVtxZ", &truevtxZ);
   regTree->SetBranchAddress("trueDirX", &truedirX);
   regTree->SetBranchAddress("trueDirY", &truedirY);
   regTree->SetBranchAddress("trueDirZ", &truedirZ);
   regTree->SetBranchAddress("hitT", &digitT);
   regTree->SetBranchAddress("hitX", &digitX);
   regTree->SetBranchAddress("hitY", &digitY);
   regTree->SetBranchAddress("hitZ", &digitZ);
   regTree->SetBranchAddress("hitType", &digitType);
   regTree->SetBranchAddress("recoStatus", &recoStatus);
   regTree->SetBranchAddress("recoVtxFOM", &recoVtxFOM);
   //regTree->SetBranchAddress("TrueInteractionType", &TrueInteractionType);
   regTree->SetBranchAddress("trueTrackLengthInMRD", &TrueTrackLengthInMrd);
   regTree->SetBranchAddress("trueTrackLengthInWater", &TrueTrackLengthInWater);
   //regTree->SetBranchAddress("TrueMomentumTransfer", &TrueMomentumTransfer);
   //regTree->SetBranchAddress("TrueMuonAngle", &TrueMuonAngle);
   regTree->SetBranchAddress("deltaVtxR", &deltaVtxR);
   regTree->SetBranchAddress("deltaAngle", &deltaAngle);
   regTree->SetBranchAddress("Pi0Count", &Pi0Count);
   regTree->SetBranchAddress("PiPlusCount", &PiPlusCount);
   regTree->SetBranchAddress("PiMinusCount", &PiMinusCount);
  
   cout<<"regTree->GetEntries(): "<<regTree->GetEntries()<<endl;
   TH1F *h1 = new TH1F("h1", "DR", 400, 0.0, 400.0);
   TH1F *h2 = new TH1F("h2", "DR for bad events", 350, 17150.0, 17500.0);
  for (Long64_t ievt=0; ievt<regTree->GetEntries(); ievt++) {
   regTree->GetEntry(ievt);

   double lambda_min = 10000000;  double lambda_max = -99999999.9; double lambda = 0;
   int totalPMTs=0; int totalLAPPDs=0; recoDWallR2=0; recoDWallZ2=0; diffDirAbs2=0;
   double lambda_vec[1100]={0.}; double digitt[1100]={0.};
   cout << "ievt:" << ievt <<endl; //printing the events number to check if all the events are being run
   //if(recoStatus == 0){ count1++;
   if(recoVtxFOM>0){ count1++;
	   cout<<"count1: "<<count1<<endl;
     //if((*TrueInteractionType == "QES - Weak[CC]") && TrueTrackLengthInMrd>0.){
      //cout<<"TrueTrackLengthInMrd: "<<TrueTrackLengthInMrd<<endl;
      if(TrueTrackLengthInMrd>0){count2++;// && Pi0Count==0 && PiPlusCount==0 && PiMinusCount==0){
        cout<<"ievt: "<<ievt<<" trueEnergy: "<<trueEnergy<<" with nhits: "<<nhits<<endl;

        //calculate diff dir with (0,0,1)
        double diffDirAbs0 = TMath::ACos(dirZ)*TMath::RadToDeg();
        //cout<<"diffDirAbs0: "<<diffDirAbs0<<endl;
        diffDirAbs2=diffDirAbs0/90.;
        double recoVtxR2 = vtxX*vtxX + vtxZ*vtxZ;//vtxY*vtxY;
        double recoDWallR = 152.4-TMath::Sqrt(recoVtxR2);
        double recoDWallZ = 198-TMath::Abs(vtxY);
        DR=TMath::Sqrt((vtxX-truevtxX)*(vtxX-truevtxX)+(vtxY-truevtxY)*(vtxY-truevtxY)+(vtxZ-truevtxZ)*(vtxZ-truevtxZ));//Calculates difference between reco and true vtx for good events
        h1->Fill(DR);

        for(int k=0; k<nhits; k++){
          //std::cout<<"k: "<<k<<", "<<digitT->at(k)<<" | "<<digitType->at(k)<<std::endl;
          digitt[k]=digitT->at(k);
          //if( (digitType->at(k)) == "PMT8inch"){ totalPMTs++; }
          //if( (digitType->at(k)) == "lappd_v0"){ totalLAPPDs++; }
          //if( (digitType->at(k)) == 0){ totalPMTs++; }
          //if( (digitType->at(k)) == 1){ totalLAPPDs++; }

           //------ Find rack Length as the distance between the reconstructed vertex last photon emission point ----/
          lambda = find_lambda(vtxX,vtxY,vtxZ,dirX,dirY,dirZ,digitX->at(k),digitY->at(k),digitZ->at(k),42.);
          if( lambda <= lambda_min ){
              lambda_min = lambda;
          }
          if( lambda >= lambda_max ){
              lambda_max = lambda;
          }
          lambda_vec[k]=lambda;
         //m_data->Stores["ANNIEEvent"]->Set("WaterRecoTrackLength",lambda_max);
        }
        std::cout<<"the track length in the water tank (1st approx) is: "<<lambda_max<<std::endl;
        cout<<"TrueTrackLengthInWater: "<<TrueTrackLengthInWater<<" TrueTrackLengthInMrd: "<<TrueTrackLengthInMrd<<endl;

       //---------------------------------
       //---- add values to tree for energy reconstruction - the values added to the BDT need to be normalised:
       ievt=ievt; //currententry;
       trueNeuE=1.*TrueNeutrinoEnergy;
       trueE=1.*trueEnergy;
       dirX2=dirX; dirY2=dirY; dirZ2=dirZ;
       vtxX2=vtxX; vtxY2=vtxY; vtxZ2=vtxZ;
       recoDWallR2      = recoDWallR/152.4;
       recoDWallZ2      = recoDWallZ/198.;
       TrueTrackLengthInMrd2 = TrueTrackLengthInMrd/200.;
       //--- 
       TrueTrackLengthInWater2 = TrueTrackLengthInWater*100.;//converting from m to cm //TrueTrackLengthInWater/500.;

        //----- write to .csv file - including variables for track length & energy reconstruction:
        for(int i=0; i<maxhits0;++i){
           csvfile<<lambda_vec[i]<<",";
        }
        for(int i=0; i<maxhits0;++i){
           csvfile<<digitt[i]<<",";
        }
        csvfile<<lambda_max<<",";
        //csvfile<<totalPMTs<<",";
	csvfile<<nhits<<",";
        csvfile<<totalLAPPDs<<",";
        csvfile<<lambda_max<<",";
        csvfile<<TrueTrackLengthInWater2<<",";
        //csvfile<<trueNeuE<<",";
        csvfile<<trueE<<",";
        csvfile<<diffDirAbs2<<",";
        csvfile<<TrueTrackLengthInMrd2<<",";
        csvfile<<recoDWallR2<<",";
        csvfile<<recoDWallZ2<<",";
        csvfile<<dirX2<<",";
        csvfile<<dirY2<<",";
        csvfile<<dirZ2<<",";
        csvfile<<vtxX2<<",";
        csvfile<<vtxY2<<",";
        csvfile<<vtxZ2<<",";
        csvfile<<truedirX<<",";
        csvfile<<truedirY<<",";
        csvfile<<truedirZ<<",";
        csvfile<<truevtxX<<",";
        csvfile<<truevtxY<<",";
        csvfile<<truevtxZ<<",";
        csvfile<<recoVtxFOM;
        //csvfile<<recoStatus<<",";
        //csvfile<<deltaVtxR<<",";
        //csvfile<<deltaAngle;
        csvfile<<'\n';
        //------------------------


     }
     else{cout<<"Negative MRD for positive recovtxFOM:"<<TrueTrackLengthInMrd<<endl;}
     }
     else{cout<<vtxX<<truevtxX<<endl;DR=TMath::Sqrt((vtxX-truevtxX)*(vtxX-truevtxX)+(vtxY-truevtxY)*(vtxY-truevtxY)+(vtxZ-truevtxZ)*(vtxZ-truevtxZ));//Calculates difference between reco and true vtx for bad events
     h2->Fill(DR);}
    }
    TFile *f = new TFile("histos.root","new");
    h1->Write();
    h2->Write();
    f->Close();
   input->Close();
cout <<"Num of events with TrueTrackLengthinMRD>0 and recoVtxFOM>0"<<count2<<endl;
cout<<"Num of events with recoVtxFOM>0: "<<count1<<endl;
}


