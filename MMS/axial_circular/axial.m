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
absb[x_] = Sqrt[1 + x^2/q[x]^2];
pgrad[f_, x_, z_, y_, t_] = (D[f[x,z,y,t],y] + 1/q[x]*D[f[x,z,y,t],z])/absb[x];
d2dpar2[f_, x_, z_, y_, t_] = (D[D[f[x, z, y, t], y], y] + 2/q[x]*D[D[f[x, z, y, t], y], z] + 
     1/q[x]^2*D[D[f[x, z, y, t], z], z])/absb[x]^2;
  
xn[x_] = (x-xmin)/(xmax-xmin);  
    (*
    d2dpar2[f_, x_, z_, y_, t_] = (D[D[f[x, z, y, t], y], y] + 2/q[x]*D[D[f[x, z, y, t], y], z] + 
     1/q[x]^2*D[D[f[x, z, y, t], z], z])/absb[x]^2;
    *)


(*
Define normalised rho and MMS solution in terms of mode numbers \
given above
*)
MmsDens[x_, z_, y_, t_] = amp*Sin[2.0*Pi*kx*xn[x]]*Sin[kz*z - phz]*Cos[ky*y- phy]*Sin[omega*t - pht];
MmsUpar[x_, z_, y_, t_]=1;



pflux[x_, z_, y_, t_]=MmsDens[x, z, y, t];
Smms[x_, z_, y_, t_]=D[MmsDens[x,z,y,t],t]-d2dpar2[MmsDens,x,z,y,t];


Print["Finished MMS Terms"];

(* Set a dummy return value *)
1


EndPackage[ ]
