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
arakawa[u_, v_, r_, p_, z_, t_] = 
  D[u[x, y, z, t],x]*D[v[x, y, z, t],z] - 
   D[u[x,y,z,t],z]*D[v[x,y,z,t],x];
xn[x_] =(x-xmin)/(xmax-xmin); 
zn[z_]=(z-zmin)/(zmax-zmin);
 


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


(*
Switches and quantities for the density time evolution
*)



SolNe[x_, y_, z_, t_] = OffsetNe + ampNe*Sin[2.0*Pi*kxNe*xn[x]]*Sin[kyNe*y - phyNe]*Sin[2.0*Pi*kzNe*zn[z]-phzNe]*Sin[2.0*Pi*omegaNe*t - phtNe];
SolNVi[x_, y_, z_, t_] = ampNVi*Sin[2.0*Pi*kxNVi*xn[x]]*Sin[kyNVi*y - phyNVi]*Sin[2.0*Pi*kzNVi*zn[z]-phzNVi]*Sin[2.0*Pi*omegaNVi*t - phtNVi];
SolPe[x_, y_, z_, t_] = OffsetPe + ampPe*Sin[2.0*Pi*kxPe*xn[x]]*Sin[kyPe*y - phyPe]*Sin[2.0*Pi*kzPe*zn[z]-phzPe]*Sin[2.0*Pi*omegaPe*t - phtPe];
SolPi[x_, y_, z_, t_] = OffsetPi + ampPe*Sin[2.0*Pi*kxPi*xn[x]]*Sin[kyPi*y - phyPi]*Sin[2.0*Pi*kzPi*zn[z]-phzPi]*Sin[2.0*Pi*omegaPi*t - phtPi];
SolVort[x_, y_, z_, t_] = ampVort*Sin[2.0*Pi*kxVort*xn[x]]*Sin[kyVort*y - phyVort]*Sin[2.0*Pi*kzVort*zn[z]-phzVort]*Sin[2.0*Pi*omegaVort*t - phtVort];
SolVePsi[x_, y_, z_, t_] = ampVePsi*Sin[2.0*Pi*kxVePsi*xn[x]]*Sin[kyVePsi*y - phyVePsi]*Sin[2.0*Pi*kzVePsi*zn[z]-phzVePsi]*Sin[2.0*Pi*omegaVePsi*t - phtVePsi];

SolTe[x_, y_, z_, t_] = SolPe[x,y,z,t]/SolNe[x,y,z,t];
SolTi[x_, y_, z_, t_] = SolPi[x,y,z,t]/SolNe[x,y,z,t];
SolVi[x_, y_, z_, t_] = SolNVi[x,y,z,t]/SolNe[x,y,z,t];
SolVe[x_, y_, z_, t_] = SolVePsi[x,y,z,t]+SolVi[x,y,z,t];

SourceNe[x_, y_, z_, t_] = D[SolNe[x,y,z,t],t]\
	-rhos0*rhos0*SWNeanomalous * Danomalous*laplaceperp[SolNe,x,y,z,t]/(rhos0*rhos0*Omegaci)\
	+SWNehyper * (rhos0^4)*(hyperD/(rhos0^4 * Omegaci)) * hyperdiffusion[SolNe,x,y,z,t];
SourceNVi[x_, y_, z_, t_] = D[SolNVi[x,y,z,t],t];
SourcePe[x_, y_, z_, t_] = D[SolPe[x,y,z,t],t];
SourcePi[x_, y_, z_, t_] = D[SolPi[x,y,z,t],t];
SourceVort[x_, y_, z_, t_] = D[SolVort[x,y,z,t],t];
SourceVePsi[x_, y_, z_, t_] = D[SolVePsi[x,y,z,t],t];

Print["Finished MMS Terms"]


(* Set a dummy return value *)
1

EndPackage[ ]

