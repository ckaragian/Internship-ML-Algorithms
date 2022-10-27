import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ROOT
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (9, 7),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

#infile = "vars_Ereco_04202019.csv"
#infile = "vars_Ereco_05202019.csv"
infile = "vars_Ereco.csv"
#infile = "vars_Ereco_06082019CC0pi.csv"

filein = open(str(infile))
print("number of events: ",filein)
df00=pd.read_csv(filein)
#df00 = df000[df000['recoVtxFOM']>87.] #strict cut to avoid mis-reconstructed events
print("df00.head() ",df00.head())
#df0=df00[['TrueTrackLengthInWater','DNNRecoLength','lambda_max']]
lambdamax_test = df00['lambda_max']
test_y = df00['TrueTrackLengthInWater']
y_predicted = df00['DNNRecoLength']
TrueTrackLengthInWater = df00['TrueTrackLengthInWater'] 
trueKE = df00['trueKE']
TrueTrackLengthInMrd = 200.*df00['TrueTrackLengthInMrd'] 
#recoVtxFOM = df00['recoVtxFOM']
#deltaVtxR = df00['deltaVtxR']
#deltaAngle = df00['deltaAngle']

#print(type(test_y)," , ", type(y_predicted))

fig, ax = plt.subplots()
ax.scatter(test_y,y_predicted)
ax.plot([test_y.min(),test_y.max()],[test_y.min(),test_y.max()],'k--',lw=3)
ax.set_xlabel('MC Tracklength in water(cm)')
ax.set_ylabel('Predicted Reconstucted Length from DNN(cm)')
plt.show()
fig.savefig('recolength.png')
plt.close(fig)

fig0, ax0 = plt.subplots()
ax0.scatter(test_y,y_predicted)
ax0.plot([test_y.min(),test_y.max()],[test_y.min(),test_y.max()],'k--',lw=3)
ax0.set_xlabel('Measured True Tracklength in water')
ax0.set_ylabel('Predicted Reconstucted Length from DNN')
ax0.set_xlim(-50.,400.)
ax0.set_ylim(-50.,400.)
#plt.show()
fig0.savefig('recolength_ZOOM.png')
plt.close(fig0)

##plt.hist2d(test_y,y_predicted, bins=100, cmap='Blues')
#fig, ax = plt.subplots()
#h = ax.hist2d(test_y, y_predicted, bins=8800, cmap='Blues')
#plt.colorbar(h[3])
##cb = plt.colorbar()
##cb.set_label('counts in bin')
#ax.set_xlim(50.,200.)
#ax.set_ylim(50.,200.)
#plt.show()

fig1, ax1 = plt.subplots()
ax1.scatter(test_y,lambdamax_test)
ax1.plot([test_y.min(),test_y.max()],[test_y.min(),test_y.max()],'k--',lw=3)
ax1.set_xlabel('Measured True Tracklength in water')
ax1.set_ylabel('Predicted (lambdamax_test)')
ax1.set_xlim(-50.,400.)
ax1.set_ylim(-50.,400.)
#plt.show()
fig1.savefig('prevrecolength.png')
plt.close(fig1)

data = abs(y_predicted-test_y)
dataprev = abs(lambdamax_test-test_y)
i=0
for y in data:
   if y<(data.mean()+data.std()):
     i+=1
print(100.*i/len(data),'% of events have  ΔR<',data.mean()+data.std())
#nbins=np.arange(0,200,10)
nbins=np.arange(0.,400.,5)
##n, bins, patches = plt.hist(data, 100, alpha=1,normed='true')
fig2,ax2=plt.subplots(ncols=1, sharey=False)#, figsize=(8, 6))
f0=ax2.hist(data, nbins, histtype='step', fill=False, color='blue',alpha=0.75) 
f1=ax2.hist(dataprev, nbins, histtype='step', fill=False, color='red',alpha=0.75)
#ax.set_xlim(0.,200.)
ax2.set_xlabel('$\Delta R = |L_{Reco}-L_{MC}|$ [cm]')
ax2.legend(('NEW','Previous'))
ax2.xaxis.set_ticks(np.arange(0., 425., 25))
ax2.tick_params(axis='x', which='minor', bottom=False)
##ax.set_ylabel('Number of Entries [%]')
##ax.xaxis.set_label_coords(0.95, -0.08)
##ax.yaxis.set_label_coords(-0.1, 0.71)

#from scipy.stats import norm
## Fit a normal distribution to the data:
##mu, std = norm.fit(data)

xmin, xmax = plt.xlim()
##x = np.linspace(xmin, xmax, 100)
##p = norm.pdf(x, mu, std)
##plt.plot(x, p, 'k', linewidth=2)
title = "mean = %.2f, std = %.2f, Prev: mean = %.2f, std = %.2f " % (data.mean(), data.std(),dataprev.mean(), dataprev.std())
plt.title(title)
plt.show()
fig2.savefig('resol_distr2l_WCSim.png')
plt.close(fig2)

data1 = 100.*abs(y_predicted-test_y)/test_y
dataprev1 = 100.*abs(lambdamax_test-test_y)/test_y
#nbins=np.arange(0,200,10)
nbins=np.arange(0.,400.,5)
##n, bins, patches = plt.hist(data, 100, alpha=1,normed='true')
fig3,ax3=plt.subplots(ncols=1, sharey=False)#, figsize=(8, 6))
f00=ax3.hist(data1, nbins, histtype='step', fill=False, color='blue',alpha=0.75) 
f11=ax3.hist(dataprev1, nbins, histtype='step', fill=False, color='red',alpha=0.75)
#ax.set_xlim(0.,200.)
ax3.set_xlabel('$\Delta R = |L_{Reco}-L_{MC}|/L_{MC}$')
ax3.legend(('NEW','Previous'))
ax3.xaxis.set_ticks(np.arange(0., 425., 25))
ax3.tick_params(axis='x', which='minor', bottom=False)
##ax.set_ylabel('Number of Entries [%]')
##ax.xaxis.set_label_coords(0.95, -0.08)
##ax.yaxis.set_label_coords(-0.1, 0.71)

#from scipy.stats import norm
## Fit a normal distribution to the data:
##mu, std = norm.fit(data)

xmin, xmax = plt.xlim()
##x = np.linspace(xmin, xmax, 100)
##p = norm.pdf(x, mu, std)
##plt.plot(x, p, 'k', linewidth=2)
title = "mean = %.2f, std = %.2f, Prev: mean = %.2f, std = %.2f " % (data1.mean(), data1.std(),dataprev1.mean(), dataprev1.std())
plt.title(title)
plt.show()
fig3.savefig('resol_distr2l_WCSimpercent.png')
plt.close(fig3)

#fig1, ax1 = plt.subplots()
#ax1.scatter(data, lambdamax_test)
#ax1.set_xlabel('$\Delta R = L_{Reco}-L_{MC}$ [cm]')
#ax1.set_ylabel('deltaVtxR')
#ax1.set_xlim(-50.,2000.)
#ax1.set_ylim(-50.,2000.)
#plt.show()
#deltaVtxR
#deltaAngle
print("--------- Reco Events Profile --------")
count_bad = 0
count_med = 0
count_good = 0
for i,r in data.items():
    if r>=30.:
       #print("data: ",round(r,2)," recoVtxFOM: ", round(recoVtxFOM[i],2)," deltaVtxR: ",round(deltaVtxR[i],2)," deltaAngle: ",round(deltaAngle[i],2))
       count_bad +=1    
    if r>=15. and r<30.:
       count_med +=1
    if r<15.:
       count_good +=1
print("Percentage of good events(<15cm): ", 100.*count_good/len(data))
print("Percentage of med events(15cm<=r<30cm): ", 100.*count_med/len(data))
print("Percentage of bad events(r>=30cm): ", 100.*count_bad/len(data))

print("checking..."," len(test_y): ",len(test_y)," len(y_predicted): ", len(y_predicted))

canvas = ROOT.TCanvas()
canvas.cd(1)
th2f = ROOT.TH2F("True_RecoLength", "; MC Track Length [cm]; Reconstructed Track Length [cm]", 50, 0, 400., 50, 0., 400.)
for i in range(len(test_y)):
    th2f.Fill(test_y[i], y_predicted[i])
line = ROOT.TLine(0.,0.,400.,400.)
th2f.SetStats(0)
th2f.Draw("ColZ")
line.SetLineColor(2)
canvas.Draw()
line.Draw("same")
canvas.SaveAs("MClength_newrecolength.png")

#---check energy dependence with track length:
canvas = ROOT.TCanvas()
canvas.cd(1)
th2f = ROOT.TH2F("True_RecoLength", "; E_{MC,muon} [MeV]; MC Track Length in Water [cm]", 20, 0, 2000., 100, 0., 400.)
for i in range(len(trueKE)):
    th2f.Fill(trueKE[i], TrueTrackLengthInWater[i])
#line = ROOT.TLine(0.,0.,400.,400.)
th2f.SetStats(0)
th2f.Draw("ColZ")
#line.SetLineColor(2)
canvas.Draw()
#line.Draw("same")
canvas.SaveAs("Emu_MClengthwater.png")

canvas = ROOT.TCanvas()
canvas.cd(1)
th2f = ROOT.TH2F("True_RecoLength", "; E_{MC,muon} [MeV]; MC Track Length in MRD [cm]", 20, 0, 2000., 100, 0., 400.)
for i in range(len(trueKE)):
    th2f.Fill(trueKE[i], TrueTrackLengthInMrd[i])
#line = ROOT.TLine(0.,0.,400.,400.)
th2f.SetStats(0)
th2f.Draw("ColZ")
#line.SetLineColor(2)
canvas.Draw()
#line.Draw("same")
canvas.SaveAs("Emu_MClengthMRD.png")


#c1 = ROOT.TCanvas()
#c1.cd(1)
#th2f = ROOT.TH2F("data_scatter", ";$\Delta R = L_{Reco}-L_{MC}$ [cm]; var", 100, 0, 2000., 100, -1000., 100.)
#for i,r in data.items():
#    print("r",r,",",deltaAngle[i])
#    th2f.Fill(r,deltaAngle[i])
#c1.Draw()
#c1.SaveAs("scatter.png")

#write in output .root file:
#outputFile = ROOT.TFile.Open("output_length.root", "RECREATE")
#outputTuple = ROOT.TNtuple("tuple", "tuple", "test_y:y_predicted:recoVtxFOM:deltaVtxR:deltaAngle")
#for i in range(len(test_y)):
#    outputTuple.Fill(test_y[i],y_predicted[i],recoVtxFOM[i],deltaVtxR[i],deltaAngle[i])
#outputTuple.Write()
#outputFile.Close()
