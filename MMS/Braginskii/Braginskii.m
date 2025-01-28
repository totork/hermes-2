(* ::Package:: *)

(* ::Package:: *)
(**)


BeginPackage["BraginskiiMMSSlab`"]

(* To run the package, execute the following line in a notebook
<< BraginskiiMMS`
 *)

(* BraginskiiMMS::usage =
        "Provides MMS solution and source for the Braginskii model in slab geometry
        - Only Dirichlet boundary conditions in the perpendicular direction are considered *)


(* BraginskiiMMS[] :=  *)

Print["Computing MMS Terms"];


(*
define operators and normalised coordinates
*)
curvature[f_, x_, y_, z_, t_] = -2*D[f[x,y,z,t],y];
laplaceperp[f_, x_, y_, z_, t_] = D[f[x,y,z,t],{x,2}] + D[f[x,y,z,t],{z,2}];
hyperdiffusion[f_, x_, y_, z_, t_] = D[f[x,y,z,t],{x,4}] + D[f[x,y,z,t],{z,4}];
numericaldiffusion[f_, x_, y_, z_, t_] = D[f[x,y,z,t],{y,4}];
gradpar[f_, x_, y_, z_, t_] = D[f[x,y,z,t],{y,1}];
divpar[f_, x_, y_, z_, t_] = D[f[x,y,z,t],{y,1}];
arakawa[u_, v_, x_, y_, z_, t_] = 
  D[u[x, y, z, t],x]*D[v[x, y, z, t],z] - 
   D[u[x,y,z,t],z]*D[v[x,y,z,t],x];
xn[x_] =(x-xmin)/(xmax-xmin); 
zn[z_]=(z-zmin)/(zmax-zmin);
divparkgradpar[k_, f_, x_, y_, z_, t_] = D[k[x,y,z,t]*D[f[x,y,z,t],{y,1}], {y,1}];
(*divparkgradpar[k_, f_, x_, y_, z_, t_] := k[x,y,z,t]*D[f[x,y,z,t],{y,2}];*)
(*divparkgradpar[k_, f_, x_, y_, z_, t_] = D[f[x,y,z,t],{y,2}];*)
divperp[f_, x_, y_, z_, t_] = D[f[x,y,z,t],{x,1}] + D[f[x,y,z,t],{z,1}];
gradperp[f_, x_, y_, z_, t_] = D[f[x,y,z,t],{x,1}] + D[f[x,y,z,t],{z,1}];


 divagradperp[k_, f_, x_, y_, z_, t_] = D[k[x,y,z,t]*D[f[x,y,z,t],x],x]+D[k[x,y,z,t]*D[f[x,y,z,t],z],z];
 divabvec[a_, b_, x_, y_, z_, t_] = D[a[x,y,z,t]*b[x,y,z][[1]],x] + D[a[x,y,z,t]*b[x,y,z][[3]],z];


(*
Quantities for the normalization of the code

*)
Tnorm = 20;
Nnorm = 10^19;
Bnorm = 1.0;
AA = 2.0;
Mp=1.672621*(10^-27);
Me=9.1093*(10^-31);
qe=1.602176*(10^-19);
Cs0 = Sqrt[(qe*Tnorm)/(AA*Mp)];
Omegaci = qe*Bnorm/(AA*Mp);
rhos0=Cs0/Omegaci;
mime = AA * Mp / (Me);
memi = Me/(AA*Mp);
lambdaei=24.0-Log[Sqrt[Nnorm/(10^6)]/Tnorm];
taue0 = 1. / (2.91 * (10^-6) * (Nnorm / (10^6)) * lambdaei * (Tnorm^(-1.5)));

lambdaii = 23.0-Log[Sqrt[2.0*Nnorm/(10^6)]/(Tnorm^1.5)];
taui0 = Sqrt[AA]/(4.78 * 10^-8 * (Nnorm/(10^6)) * lambdaii * (Tnorm^(-1.5)));

B[x_, y_, z_, t_] = 1.5-x;
Bvec[x_, y_, z_] = {0.0, 1.5-x,0.0};
SqrtB[x_,y_,z_, t_] = Sqrt[B[x,y,z,t]];
curlBvec[x_, y_, z_] = Curl[Bvec[x,y,z]/Norm[Bvec[x,y,z]], {x,y,z}];
logB[x_, y_, z_,t_] = Log[B[x,y,z,t]];
curvature[f_, x_, y_, z_, t_] = -2.0*arakawa[B,f,x,y,z,t]/(B[x,y,z,t]^2);
(*Das minuszeichenb kommt durch in brendans paper nicht vor, in gleichung A8 fehlt das*)



(*
Switches and quantities for the density time evolution
*)



SolNe[x_, y_, z_, t_] = OffsetNe + ampNe*Sin[2.0*Pi*kxNe*xn[x]]*Sin[kyNe*y - phyNe]*Sin[2.0*Pi*kzNe*zn[z]-phzNe]*Sin[2.0*Pi*omegaNe*t - phtNe];
SolNVi[x_, y_, z_, t_] = ampNVi*Sin[2.0*Pi*kxNVi*xn[x]]*Sin[kyNVi*y - phyNVi]*Sin[2.0*Pi*kzNVi*zn[z]-phzNVi]*Sin[2.0*Pi*omegaNVi*t - phtNVi];
SolPe[x_, y_, z_, t_] = OffsetPe + ampPe*Sin[2.0*Pi*kxPe*xn[x]]*Sin[kyPe*y - phyPe]*Sin[2.0*Pi*kzPe*zn[z]-phzPe]*Sin[2.0*Pi*omegaPe*t - phtPe];
SolPi[x_, y_, z_, t_] = OffsetPi + ampPi*Sin[2.0*Pi*kxPi*xn[x]]*Sin[kyPi*y - phyPi]*Sin[2.0*Pi*kzPi*zn[z]-phzPi]*Sin[2.0*Pi*omegaPi*t - phtPi];
SolVort[x_, y_, z_, t_] = ampVort*Sin[2.0*Pi*kxVort*xn[x]]*Sin[kyVort*y - phyVort]*Sin[2.0*Pi*kzVort*zn[z]-phzVort]*Sin[2.0*Pi*omegaVort*t - phtVort];
SolVePsi[x_, y_, z_, t_] = ampVePsi*Sin[2.0*Pi*kxVePsi*xn[x]]*Sin[kyVePsi*y - phyVePsi]*Sin[2.0*Pi*kzVePsi*zn[z]-phzVePsi]*Sin[2.0*Pi*omegaVePsi*t - phtVePsi];

SolTe[x_, y_, z_, t_] = SolPe[x,y,z,t]/SolNe[x,y,z,t];
SolTi[x_, y_, z_, t_] = SolPi[x,y,z,t]/SolNe[x,y,z,t];
SolVi[x_, y_, z_, t_] = SolNVi[x,y,z,t]/SolNe[x,y,z,t];
SolVe[x_, y_, z_, t_] = SolVePsi[x,y,z,t]+SolVi[x,y,z,t];
SolJpar[x_, y_, z_, t_] = SolNVi[x,y,z,t]-SolNe[x,y,z,t]*SolVe[x,y,z,t];

SolNeVi[x_, y_, z_, t_] = SolNe[x,y,z,t]*SolVi[x,y,z,t];
SolPepPi[x_, y_, z_, t_] = SolPe[x,y,z,t]+SolPi[x,y,z,t];
SolPeVe[x_, y_, z_, t_] = SolPe[x,y,z,t]*SolVe[x,y,z,t];
SolPiVi[x_, y_, z_, t_] = SolPi[x,y,z,t]*SolVi[x,y,z,t];
SolTeJpar[x_, y_, z_, t_] = SolTe[x,y,z,t]*SolJpar[x,y,z,t];
SolNViVi[x_, y_, z_, t_] = SolNVi[x,y,z,t] * SolVi[x,y,z,t];

taue[x_, y_, z_, t_] = (taue0 * (Cs0/rhos0) * (SolTe[x,y,z,t]^(1.5)))/SolNe[x,y,z,t];
taui[x_, y_, z_, t_] = (taui0 * (Cs0/rhos0) * (SolTi[x,y,z,t]^(1.5)))/SolNe[x,y,z,t];

kappaepar[x_, y_, z_, t_] = 3.16 * mime * SolTe[x,y,z,t] * SolNe[x,y,z,t] * taue[x,y,z,t];
kappaipar[x_, y_, z_, t_] = 3.9 * SolTi[x,y,z,t] * SolNe[x,y,z,t] * taui[x,y,z,t];

PitauidivB[x_, y_, z_, t_] = SolPi[x,y,z,t]*taui[x,y,z,t]/B[x,y,z];
B12Vi[x_, y_, z_, t_] = SqrtB[x,y,z] * SolVi[x,y,z,t];
SolViDanomalous[x_, y_, z_, t_] = SolVi[x,y,z,t]*Danomalous;
SolNenuanomalous[x_, y_, z_, t_] = SolNe[x,y,z,t] * nuanomalous;
DanomalousVi[x_, y_, z_, t_] = Danomalous * SolVi[x,y,z,t];
nuanomalousNe[x_, y_, z_, t_] = nuanomalous * SolNe[x,y,z,t];
DanomalousTe[x_, y_, z_, t_] = Danomalous * SolTe[x,y,z,t];
chianomalousNe[x_, y_, z_, t_] = chianomalous * SolNe[x,y,z,t];
DanomalousTi[x_, y_, z_, t_] = Danomalous * SolTi[x,y,z,t];
(*divagradperp[SolViDanomalous,SolNe,x,y,z,t]*)
Vmage[x_, y_, z_, t_] = -SolTe[x,y,z,t]*curlBvec[x,y,z];




SourceNe[x_, y_, z_, t_] = D[SolNe[x,y,z,t],t]\
	-rhos0*rhos0*SWNeanomalous * Danomalous*laplaceperp[SolNe,x,y,z,t]/(rhos0*rhos0*Omegaci)\
	+SWNehyper * (rhos0^4)*(hyperD/(rhos0^4 * Omegaci)) * hyperdiffusion[SolNe,x,y,z,t]\
	+SWNenumdiff * (rhos0^4)*(numD/(rhos0^4 * Omegaci)) * numericaldiffusion[SolNe,x,y,z,t]\
	+SWNeparflow * rhos0 * gradpar[SolNeVi,x,y,z,t]\
	(*+SWNemag * (rhos0^2) * divabvec[SolNe,Vmage,x,y,z,t]*)\
	-SWNemag * (rhos0^2) * curvature[SolPe,x,y,z,t];
SourceNVi[x_, y_, z_, t_] = D[SolNVi[x,y,z,t],t]\
	+SWNViparpressure*rhos0*gradpar[SolPepPi,x,y,z,t]\
	+SWNVihyper * (rhos0^4)*(hypernu/(rhos0^4 * Omegaci)) * hyperdiffusion[SolNVi,x,y,z,t]\
	+SWNVinumdiff * (rhos0^4)*(numnu/(rhos0^4 * Omegaci)) * numericaldiffusion[SolNVi,x,y,z,t]\
	-SWNViparviscos * (rhos0^2) * 1.28 * SqrtB[x,y,z] * divparkgradpar[PitauidivB,B12Vi,x,y,z,t]\
	+SWNViparflow * rhos0 * divpar[SolNViVi,x,y,z,t]\
	-SWNVianomalous * (divagradperp[DanomalousVi,SolNe,x,y,z,t] + divagradperp[nuanomalousNe,SolVi,x,y,z,t])*(rhos0^2)/(rhos0*rhos0*Omegaci);
SourcePe[x_, y_, z_, t_] = D[SolPe[x,y,z,t],t]\
	- SWPeconduction*(2.0/3.0)*divparkgradpar[kappaepar,SolTe,x,y,z,t]*(rhos0^2)\
	+SWPenumdiff * (rhos0^4)*(numchi/(rhos0^4 * Omegaci)) * numericaldiffusion[SolPe,x,y,z,t]\
	+SWPehyper * (rhos0^4)*(hyperchi/(rhos0^4 * Omegaci)) * hyperdiffusion[SolPe,x,y,z,t]\
	+SWPeparflow * rhos0*(divpar[SolPeVe,x,y,z,t] + (2.0/3.0)*SolPe[x,y,z,t]*divpar[SolVe,x,y,z,t])\
	-SWPethermalcurrent * rhos0 * (0.71*2.0/3.0) * divpar[SolTeJpar,x,y,z,t]\
	+SWPethermalforce * rhos0 * (0.71*2.0/3.0) * SolJpar[x,y,z,t]*gradpar[SolTe,x,y,z,t]\
	-SWPeanomalous * (2.0/3.0) * (divagradperp[DanomalousTe,SolNe,x,y,z,t] + divagradperp[chianomalousNe,SolTe,x,y,z,t])*(rhos0^2)/(rhos0*rhos0*Omegaci);
SourcePi[x_, y_, z_, t_] = D[SolPi[x,y,z,t],t]\
	- SWPiconduction * (2.0/3.0) * divparkgradpar[kappaipar,SolTi,x,y,z,t] * (rhos0^2)\
	+SWPiparflow * rhos0*(divpar[SolPiVi,x,y,z,t] + (2.0/3.0)*SolPi[x,y,z,t]*divpar[SolVi,x,y,z,t])\
	+SWPinumdiff * (rhos0^4)*(numchi/(rhos0^4 * Omegaci)) * numericaldiffusion[SolPi,x,y,z,t]\
	+SWPihyper * (rhos0^4)*(hyperchi/(rhos0^4 * Omegaci)) * hyperdiffusion[SolPi,x,y,z,t]\
	-SWPianomalous * (2.0/3.0) * (divagradperp[DanomalousTi,SolNe,x,y,z,t] + divagradperp[chianomalousNe,SolTi,x,y,z,t])*(rhos0^2)/(rhos0*rhos0*Omegaci);
SourceVort[x_, y_, z_, t_] = D[SolVort[x,y,z,t],t];
SourceVePsi[x_, y_, z_, t_] = D[SolVePsi[x,y,z,t],t];

Print["Finished MMS Terms"]


(* Set a dummy return value *)
1

EndPackage[ ]

