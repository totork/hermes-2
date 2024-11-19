(* ::Package:: *)

BeginPackage["axial`"]

(* To run the package, execute the following line in a notebook
<<axial.m
 *)

(* 
"Provides MMS solution and source for the anisotropic diffusion model in circular geometry
- Only Dirichlet boundary conditions in the perpendicular direction is considered
- Parallel boundaries (Penalisation) are not taken into account 
- MMS solution is firstly is prescribed in cylindrical coordinates 
(r,p,z) and afterwards a transformed to Cartesian coordinates (x,y,phi) used in GRILLIX 
r = sqrt(x^2+y^2);  p = arctan(y/x);  z = phi, t = time
- Paramters of model are: chi_par, chi_perp, safety factor q, and limiting flux surfaces rmin and rmax"
*)


Print["Computing MMS Terms"];

(*
define x,y and parallel derivatives in terms of r,p,z derivative \
and normalised radial coordinate rn
*)
absb[r_] = Sqrt[1 + r^2/q[r]^2];
pgrad[f_, r_, p_, z_, t_] = (D[f[r,p,z,t],z] + 1/q[r]*D[f[r,p,z,t],p])/absb[r];


(*
Define normalised rho and MMS solution in terms of mode numbers \
given above
*)
MmsDens[r_, p_, z_, t_] = amp*Sin[kr*r]*Sin[kp*p - php]*Cos[kz*z - phz]*  Sin[omega*t - pht]


Smms[r_, p_, z_, t_]=D[MmsDens[r,p,z,t],t]-pgrad[MmsDens[r,p,z,t],r,p,z,t]


Print["Finished MMS Terms"];

(* Set a dummy return value *)
1


EndPackage[ ]
