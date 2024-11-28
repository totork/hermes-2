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
ddx[f_, r_, p_, z_, t_] = D[f[r, p, z, t], r];
ddy[f_, r_, p_, z_, t_] = D[f[r, p, z, t], r]*Sin[p] + D[f[r, p, z, t], p]*Cos[p]/r;
d2dx2[f_, r_, p_, z_, t_] = D[ddx[f, r, p, z, t], r];
d2dy2[f_, r_, p_, z_, t_] = D[ddy[f, r, p, z, t], r]*Sin[p] +  D[ddy[f, r, p, z, t], p]*Cos[p]/r;



LaplacePerpMmsSol[f_,r_, p_, z_, t_] =  d2dx2[f, r, p, z, t] + d2dy2[f, r, p, z, t];

LaplacePerp[f_,r_,p_,z_,t_] = D[f[r,p,z,t],{r,2}] + D[f[r,p,z,t],r]/r + 1.0/(r*r)*D[f[r,p,z,t],{p,2}];

d2dpar2[f_, x_, z_, y_, t_] = Dpar * ((D[D[f[x, z, y, t], y], y] + 2/q[x]*D[D[f[x, z, y, t], y], z] + 
     1/q[x]^2*D[D[f[x, z, y, t], z], z])/absb[x]^2);
  
xn[x_] = (x-xmin)/(xmax-xmin);  

Arakawa[a_,b_,r_,p_,z_,t_]=D[a[r,p,z,t],r]*D[b[r,p,z,t],p]-D[a[r,p,z,t],p]*D[b[r,p,z,t],r];
Bx[r_,p_,z_,t_]=-r*Sin[p] / q[r];
Bz[r_,p_,z_,t_]=r*Cos[p] / q[r];
Bmag[r_,p_,z_,t_]=Sqrt[By^2 + Bx[r,p,z,t]^2+Bz[r,p,z,t]^2];


(*
Define normalised rho and MMS solution in terms of mode numbers \
given above
*)
MmsPhi[x_, z_, y_, t_] = ampPhi*Sin[2.0*Pi*kxPhi*xn[x]]*Sin[kzPhi*z - phzPhi]*Cos[kyPhi*y- phyPhi]*Sin[omegaPhi*t - phtPhi];
MmsApar[x_, z_, y_, t_] = ampApar*Sin[2.0*Pi*kxApar*xn[x]]*Sin[kzApar*z - phzApar]*Cos[kyApar*y- phyApar]*Sin[omegaApar*t - phtApar];
MmsU[x_,z_,y_,t_]=1.0/Bmag[x,z,y,t]*LaplacePerp[MmsPhi,x,z,y,t];
MmsJpar[x_,z_,y_,t_]=-LaplacePerp[MmsApar,x,z,y,t]



(*Smms[x_, z_, y_, t_]=D[MmsDens[x,z,y,t],t]-Dperp * LaplacePerp[MmsDens,x,z,y,t]-d2dpar2[MmsDens,x,z,y,t];*)
JpardivBmag[x_,z_,y_,t_]=MmsJpar[x,z,y,t]/Bmag[x,z,y,t];
MmsUSource[x_,z_,y_,t_]=D[MmsU[x,z,y,t],t] + SwitchUExB / Bmag[x,z,y,t]* Arakawa[MmsPhi,MmsU,x,z,y,t] - SwitchUDivpar * Bmag[x,z,y,t]^2 * pgrad[JpardivBmag,x,z,y,t]-mu*LaplacePerp[MmsU,x,z,y,t]
MmsAparSource[x_,z_,y_,t_]=D[MmsApar[x,z,y,t],t] + SwitchAparDivpar * pgrad[MmsPhi,x,z,y,t] + SwitchAparRes * eta * MmsJpar[x,z,y,t]





Print["Finished MMS Terms"];

(* Set a dummy return value *)
1


EndPackage[ ]
