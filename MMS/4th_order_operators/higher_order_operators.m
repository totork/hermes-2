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
xn[x_] = (x - xmin)/(xmax - xmin);
zn[z_] = (z - zmin)/(zmax - zmin);

MmsSol[x_, y_, z_, t_] = amp*Sin[twopi*kx*xn[x]]*Sin[ky*y - phy] * Cos[kz*zn[z] - phz] * Sin[omega*t - pht];





MmsSource[x_, y_, z_, t_] = (D[MmsSol[x,y,z,t],t]-Dgrad*(D[MmsSol[x,y,z,t],{y,2}]));







Print["Finished MMS Terms"];

(* Set a dummy return value *)
1


EndPackage[ ]
