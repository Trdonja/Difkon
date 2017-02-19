use matrices::*;

/*
This structure stores the information needed to compute the transformation
    F: ℝ² -> ℝ², F(u) = V(r)*R(r)*(u - u₀)
and its gradient with respect to transformation parameters, if needed.
Here, u₀ = (x₀, y₀) ∈ ℝ² is the center of transformation,
    r = ‖u - u₀‖ = √((x - x₀)² + (y - y₀)²) and
    V, R: (0,∞) -> ℝ^(2⨯2)
are matrix-valued functions, constructed as described below.

The influence range of transformation is given by continuously differentiable
function λ : (0,∞) -> [0, 1] and scale parameter a > 0, which shows itself in λ(a*r).

At given r > 0, matrix V(r) is a convex combination of Vg and I,
    V(r) = λ(a*r)*Vg + (1 - λ(a*r))*I,
where I is 2⨯2 identity matrix and
    Vg = s₁*v₁*v₁ᵀ + s2*v₂*v₂ᵀ,
is a symmetric positive definite matrix, given by
    (1) orthonormal basis {v₁, v₂} for ℝ², which is standard basis {e₁, e₂}
        rotated around origin by angle ϑ and
    (2) principal stretches s₁, s₂ > 0 along principal directions v₁ and v₂.
V(r) is equivalent to its expanded form
    V(r) = λ(a*r)*((s₁ - s₂)*v₁*v₁ᵀ + (s₂ - 1)*I) + I,
where matrix representation of v₁*v₁ᵀ in standard basis is
    ⎡   cos²(ϑ)     cos(ϑ)*sin(ϑ)⎤
    ⎣cos(ϑ)*sin(ϑ)     sin²(ϑ)   ⎦.
For the purpose of efficient evaluation of values of transformation V(r), this structure
stores elements {ξᵢⱼ | i, j = 1, 2} of matrix  Ξ = (s₁ - s₂)*v₁*v₁ᵀ + (s₂ - 1)*I, which
is parametrized with ϑ, s₁ and s₂ only. The whole V(r) is additionaly parametrized with
a, x₀ and y₀ through factor λ(a*r).

Transformation R(r) at given r is a rotation by angle λ(a*r)*φ and its matrix
representation is
    ⎡cos(λ(a*r)*φ)  -sin(λ(a*r)*φ)⎤
    ⎣sin(λ(a*r)*φ)   cos(λ(a*r)*φ)⎦.
R(r) is parametrized with angle φ and a, x₀, y₀ through term λ(a*r).

The whole transformation F = F(u;θ) is parametrized with parameter vector
    θ = [x₀, y₀, ϑ, s₁, s₂, φ, a] from open subset of ℝ⁷.
*/
struct MapF {

    x0: f64, // x₀
    y0: f64, // y₀
    s1: f64, // principal stretch in principal direction v_1 FIXME: We dont need that stored, do we?
    s2: f64, // principal stretch in principal direction v_2 FIXME: We dont need that stored, do we?
    fi: f64, // φ, rotation for R
    a:  f64, // scale parameter
    lambda: Box<Fn(f64) -> f64>, // function λ : (0,∞) -> [0, 1]

    // Symmetric matrix Ξ, stored for efficient evaluation of V(r) = λ(a*r)*Ξ + I:
    mat_xi: SymMatrix,

    /*
    Below are stored values which are needed for efficient computation of
    gradient of transformation F with respect to parameters, ∂F(u;θ)/∂θ.
    */

    // Derivative of function λ, λ'(x) = dλ(x)/dx:
    dlambda: Box<Fn(f64) -> f64>,

    // Symmetric matrix H, which appears in ∂V/∂ϑ = λ(a*r)*H,
    //     H = (s₁ - s₂)*⎡-2*cos(ϑ)*sin(ϑ)   cos²(ϑ) - sin²(ϑ)⎤
    //                   ⎣cos²(ϑ) - sin²(ϑ)   2*cos(ϑ)*sin(ϑ) ⎦ :
    mat_h: SymMatrix,
    // Symmetric matrices 
    // P₁ = v₁*v₁ᵀ = ⎡   cos²(ϑ)     cos(ϑ)*sin(ϑ)⎤
    //               ⎣cos(ϑ)*sin(ϑ)     sin²(ϑ)   ⎦  and
    // P₂ = v₂*v₂ᵀ = ⎡   sin²(ϑ)     -cos(ϑ)*sin(ϑ)⎤
    //               ⎣-cos(ϑ)*sin(ϑ)     cos²(ϑ)   ⎦,
    // which appear in ∂V/∂sᵢ = λ(a*r)*Pᵢ, (i = 1, 2) :
    mat_p1: SymMatrix, // P₁
    mat_p2: SymMatrix // P₂
}

// Structure GradF contains partial derivatives of transformation F with respect
// to its transformation parameters x₀, y₀, ϑ, s₁, s₂, φ and a, already evaluated
// at some point u ∈ ℝ². This structure is used to compute gradient of F(u;θ) with
// respect to parameter vector
//     θ = [x₀, y₀, ϑ, s₁, s₂, φ, a] ∈ ℝ⁷.
// Gradient ∂F(u;θ)/∂θ is a linear map, which transforms any
//     h = [h1, h2, h3, h4, h5, h6, h7] ∈ ℝ⁷
// to a vector in ℝ² according to the rule
//
//     ∂F(u;θ)     ∂F(u;θ)      ∂F(u;θ)      ∂F(u;θ)      ∂F(u;θ)
//     ⎯⎯⎯⎯⎯⎯⎯⋅h = ⎯⎯⎯⎯⎯⎯⎯⋅h1 + ⎯⎯⎯⎯⎯⎯⎯⋅h2 + ⎯⎯⎯⎯⎯⎯⎯⋅h3 + ⎯⎯⎯⎯⎯⎯⎯⋅h4 +
//       ∂θ          ∂x₀          ∂y₀          ∂ϑ           ∂s₁
//                 ∂F(u;θ)      ∂F(u;θ)      ∂F(u;θ)
//               + ⎯⎯⎯⎯⎯⎯⎯⋅h5 + ⎯⎯⎯⎯⎯⎯⎯⋅h6 + ⎯⎯⎯⎯⎯⎯⎯⋅h7.
//                   ∂s₂          ∂φ           ∂a
//
struct GradF {    
    df_dx0:    Vector, // ∂F/∂x₀ = -x₀*a*λ'(a*r)/r*z + V(r)*R(r)*[-1 0]ᵀ
    df_dy0:    Vector, // ∂F/∂y₀ = -y₀*a*λ'(a*r)/r*z + V(r)*R(r)*[-1 0]ᵀ
    df_dtheta: Vector, // ∂F/∂ϑ = λ(a*r)*H*R(r)*(u - u₀)
    df_ds1:    Vector, // ∂F/∂s₁ = λ(a*r)*P₁*R(r)*(u - u₀)
    df_ds2:    Vector, // ∂F/∂s₂ = λ(a*r)*P₂*R(r)*(u - u₀)
    df_dfi:    Vector, // ∂F/∂φ = λ(a*r)*V(r)*dR(r)*(u - u₀)
    df_da:     Vector  // ∂F/∂a = λ'(a*r)*r*z
}

impl MapF {

    fn create<F>(x0: f64, y0: f64, theta: f64, s1: f64, s2: f64, fi: f64, a: f64, lambda: F, dlambda: F) -> MapF
    where F: Fn(f64) -> f64 + 'static {

        let ct = theta.cos(); // cos(ϑ)
        let st = theta.sin(); // sin(ϑ)
        let sdif = s1 - s2; // s₁ - s₂, needed in matrices Ξ and H

        // These values appear in matrices Ξ, P₁, P₂ and H:
        let ctsq = ct*ct; // cos²(ϑ)
        let stsq = st*st; // sin²(ϑ)
        let ctst = ct*st; // cos(ϑ)*sin(ϑ)
        // These values appear in matrix H:
        let hdiag = sdif*2.0*ctst;      // 2*(s₁ - s₂)*cos(ϑ)*sin(ϑ)
        let hoffd = sdif*(ctsq - stsq); // (s₁ - s₂)*(cos²(ϑ) - sin²(ϑ))

        let mat_xi = SymMatrix { // matrix Ξ
                element_11: sdif*ctsq + s2 - 1.0, // ξ₁₁
                element_12: sdif*ctst,            // ξ₁₂ = ξ₂₁
                element_22: sdif*stsq + s2 - 1.0  // ξ₂₂
        };

        MapF {
            x0: x0,
            y0: y0,
            s1: s1, // FIXME: Do we need this?
            s2: s2, // FIXME: Do we need this?
            fi: fi,
            a: a,
            lambda: Box::new(lambda),

            mat_xi: mat_xi,

            dlambda: Box::new(dlambda),

            mat_h: SymMatrix {
                element_11: -hdiag,
                element_12: hoffd,
                element_22: hdiag
            },
            mat_p1: SymMatrix {
                element_11: ctsq,
                element_12: ctst,
                element_22: stsq
            }, 
            mat_p2: SymMatrix {
                element_11: stsq,
                element_12: -ctst,
                element_22: ctsq
            }
        }
    }

    // For given u∈ℝ² and F(.;θ), represented by struct MapF, this function
    // evaluates the value F(u;θ) (in type Vector).
    // FIXME: Popravi ime funkcije oz. uskladi ime s funkcijo, ki vrača tudi gradient.
    fn eval(&self, x: f64, y: f64) -> Vector {

        //  d = [xd, yd]ᵀ := [x - x₀, y - y₀]ᵀ = u - u₀
        let d = Vector{x: x - self.x0, y: y - self.y0};

        // compute λ(a*r)
        let r = (d.x*d.x + d.y*d.y).sqrt();
        let lambda = (*(self.lambda))(self.a*r);

        // compute elements of symetric matrix V(r) = λ(a*r)*Ξ + I
        //     V(r) = λ(a*r)*⎡ξ₁₁ ξ₁₂⎤ + ⎡1 0⎤
        //                   ⎣ξ₁₂ ξ₂₂⎦   ⎣0 1⎦
        let mat_v = SymMatrix {
            element_11: lambda*self.mat_xi.element_11 + 1.0,
            element_12: lambda*self.mat_xi.element_12,
            element_22: lambda*self.mat_xi.element_22 + 1.0
        };
        // compute elements of ez-matrix R(r)
        //      ⎡cos(λ(a*r)*φ)  -sin(λ(a*r)*φ)⎤
        //      ⎣sin(λ(a*r)*φ)   cos(λ(a*r)*φ)⎦
        let lfi = lambda*self.fi; // λ(a*r)*φ
        let mat_r = EzMatrix {
            element_11: lfi.cos(), // cos(λ(a*r)*φ)
            element_21: lfi.sin()  // sin(λ(a*r)*φ)
        };

        // final result F(u) = V(r)*R(r)*(u - u₀)
        &mat_v * &(&mat_r * &d)
    }

    // For given u∈ℝ² and F(.;θ), represented by struct MapF, this function
    // evaluates the value F(u;θ) (in type Vector) and gradient ∂F(u;θ)/∂θ
    // (in type GradientF).
    fn evaluate(&self, x: f64, y: f64) -> (Vector, GradF) {

        //  d = [xd, yd]ᵀ := [x - x₀, y - y₀]ᵀ = u - u₀
        let d = Vector{x: x - self.x0, y: y - self.y0};

        // compute λ(a*r) and λ'(a*r)
        let r = (d.x*d.x + d.y*d.y).sqrt();
        let ar = self.a*r;
        let lambda = (*(self.lambda))(ar);   // λ(a*r)
        let dlambda = (*(self.dlambda))(ar); // λ'(a*r)

        // ______ Computation of value F(u;θ) ______
        
        // compute elements of symetric matrix V(r) = λ(a*r)*Ξ + I
        //     V(r) = λ(a*r)*⎡ξ₁₁ ξ₁₂⎤ + ⎡1 0⎤
        //                   ⎣ξ₁₂ ξ₂₂⎦   ⎣0 1⎦
        let mat_v = SymMatrix {
            element_11: lambda*self.mat_xi.element_11 + 1.0,
            element_12: lambda*self.mat_xi.element_12,
            element_22: lambda*self.mat_xi.element_22 + 1.0
        };
        // compute elements of ez-matrix R(r)
        //      ⎡cos(λ(a*r)*φ)  -sin(λ(a*r)*φ)⎤
        //      ⎣sin(λ(a*r)*φ)   cos(λ(a*r)*φ)⎦
        let lfi = lambda*self.fi; // λ(a*r)*φ
        let mat_r = EzMatrix {
            element_11: lfi.cos(), // cos(λ(a*r)*φ)
            element_21: lfi.sin()  // sin(λ(a*r)*φ)
        };
        // compute product matrix V(r)*R(r)
        let mat_vr = &mat_v*&mat_r;
        // Final result F(u) = V(r)*R(r)*(u - u₀) is computed and returned at the
        // bottom of this function.

        // ______ Computatuion of gradient ∂F(u;θ)/∂θ ______            
        
        // ez-matrix dR(r)
        //    ⎡-sin(λ(a*r)*φ)  -cos(λ(a*r)*φ)⎤
        //    ⎣ cos(λ(a*r)*φ)  -sin(λ(a*r)*φ)⎦
        let mat_dr = EzMatrix {
            element_11: -mat_r.element_21, // -sin(λ(a*r)*φ)
            element_21: mat_r.element_11   // cos(λ(a*r)*φ)
        };
        let rd = &mat_r*&d; // R*(u - u₀)
        let drd = &mat_dr*&d; // dR(λ(a*r))*(u - u₀)
        // z = Ξ*R(r)*(u - u₀) + φ*V(r)*dR(r)*(u - u₀)
        let z = &(self.mat_xi)*&rd + self.fi*&(&mat_v*&drd);
        let jupa = -self.a*dlambda/r; // -a*λ'(a*r)/r
        let lrd = lambda*&rd; // λ(a*r)*R*(u - u₀)
        
        // Computation of partial derivatives:

        let grad_f = GradF {        
            // ∂F/∂x₀ = -x₀*a*λ'(a*r)/r*z + V(r)*R(r)*[-1 0]ᵀ
            df_dx0:    (self.x0*jupa)*&z + Vector{x: -mat_vr.element_11, y: -mat_vr.element_21},
            // ∂F/∂y₀ = -y₀*a*λ'(a*r)/r*z + V(r)*R(r)*[-1 0]ᵀ
            df_dy0:    (self.y0*jupa)*&z + Vector{x: -mat_vr.element_12, y: -mat_vr.element_22},
            // ∂F/∂ϑ = λ(a*r)*H*R(r)*(u - u₀)
            df_dtheta: &(self.mat_h)*&lrd,
            // ∂F/∂s₁ = λ(a*r)*P₁*R(r)*(u - u₀)
            df_ds1:    &(self.mat_p1)*&lrd,
            // ∂F/∂s₂ = λ(a*r)*P₂*R(r)*(u - u₀)
            df_ds2:    &(self.mat_p2)*&lrd,
            // ∂F/∂φ = λ(a*r)*V(r)*dR(r)*(u - u₀)
            df_dfi:    lambda*&(&mat_v*&drd),
            // ∂F/∂a = λ'(a*r)*r*z
            df_da:     (dlambda*r)*&z
        };

        // ______ Final result ______ 

        (&mat_vr*&d, grad_f)
    }

// TODO: impl GradF, ki ima funkcijo, ki za tabelo parametrov evaluira gradient.

}
