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
ddy[f_, r_, p_, z_, t_] = D[f[r, p, z, t], p]/r;
d2dx2[f_, r_, p_, z_, t_] = D[ddx[f, r, p, z, t], r];
d2dy2[f_, r_, p_, z_, t_] = D[ddy[f, r, p, z, t], r]*Sin[p] +  D[ddy[f, r, p, z, t], p]*Cos[p]/r;

arakawa[u_, v_, r_, p_, z_, t_] = 
  ddx[u, r, p, z, t]*ddy[v, r, p, z, t] - 
   ddy[u, r, p, z, t]*ddx[v, r, p, z, t];
bracketoperator[f_, g_, r_, p_, z_, t_] = -arakawa[g,f,r,p,z,t];
ExBoperator[f_, r_, p_, z_, t_] = -arakawa[MmsPhi,f,r,p,z,t];


LaplacePerpMmsSol[f_,r_, p_, z_, t_] =  d2dx2[f, r, p, z, t] + d2dy2[f, r, p, z, t];

LaplacePerp[f_,r_,p_,z_,t_] = D[f[r,p,z,t],{r,2}] + D[f[r,p,z,t],r]/r + 1.0/(r*r)*D[f[r,p,z,t],{p,2}];
divagradperp[a_, f_, r_, p_, z_, t_] = D[a[r,p,z,t],r]*D[f[r,p,z,t],r] + a[r,p,z,t]*D[f[r,p,z,t],{r,2}]+1.0/(r)*a[r,p,z,t]*D[f[r,p,z,t],r]\
									+1.0/(r^2)*(D[a[r,p,z,t],p]*D[f[r,p,z,t],p] + a[r,p,z,t]*D[f[r,p,z,t],{p,2}]);
									
									
d2dpar2[f_, x_, z_, y_, t_] = Dpar[x,z,y,t] * ((D[D[f[x, z, y, t], y], y] + 2/q[x]*D[D[f[x, z, y, t], y], z] + 
     1/q[x]^2*D[D[f[x, z, y, t], z], z])/absb[x]^2);
  
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
MmsPhi[x_, y_, z_, t_] = ampphi*Sin[2.0*Pi*kxphi*xn[x]]*Sin[kzphi*z]*Cos[kyphi*y]*Sin[omega*t - pht];
MmsDelp2Ne[x_, z_, y_, t_] = LaplacePerp[MmsDens, x, z, y, t];


pflux[x_, z_, y_, t_]=MmsDens[x, z, y, t];
qcond[x_, z_, y_, t_] = Dpar[x,z,y,t]*pgrad[MmsDens,x,z,y,t];
(*Smms[x_, z_, y_, t_]=D[MmsDens[x,z,y,t],t]-Dperp * LaplacePerp[MmsDens,x,z,y,t]-d2dpar2[MmsDens,x,z,y,t];*)
Smms[x_, z_, y_, t_]=D[MmsDens[x,z,y,t],t]\
				-divagradperp[Dperp,MmsDens,x,z,y,t]\
				-pgrad[qcond,x,z,y,t]\
				+scaleExB*ExBoperator[MmsDens, x, z, y, t];


Print["Finished MMS Terms"];

(* Set a dummy return value *)
1


EndPackage[ ]
