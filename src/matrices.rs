use std::ops::{Add, Mul};

// Real 2⨯2 matrix
//     ⎡a11  a12⎤
//     ⎣a21  a22⎦.
pub struct Matrix {
    pub element_11: f64, // a11
    pub element_12: f64, // a12
    pub element_21: f64, // a21
    pub element_22: f64  // a22
}

// Real symetric 2⨯2 matrix
//     ⎡κ₁  τ⎤
//     ⎣τ  κ₂⎦.
pub struct SymMatrix {
    pub element_11: f64, // κ₁
    pub element_12: f64, // τ
    pub element_22: f64  // κ₂
}

// Real 2⨯2 matrix of form
//     ⎡η -ζ⎤
//     ⎣ζ  η⎦.
pub struct EzMatrix {
    pub element_11: f64, // η
    pub element_21: f64  // ζ
}

// Real 2-dimensional vector [x, y]ᵀ ∈ ℝ²
pub struct Vector {
    pub x: f64,
    pub y: f64
}

impl Add for Vector {
    type Output = Vector;
    fn add(self, other: Self) -> Vector { // Self = Vector
        Vector {
            x: self.x + other.y,
            y: self.y + other.y
        }
    }
}

impl<'a, 'b> Mul<&'b Vector> for &'a Matrix {
    type Output = Vector;
    fn mul(self, v: &'b Vector) -> Vector {
        Vector {
            x: self.element_11*v.x + self.element_12*v.y,
            y: self.element_21*v.x + self.element_22*v.y
        }
    }
}

impl <'a> Mul<&'a Vector> for f64 {
    type Output = Vector;
    fn mul(self, v: &'a Vector) -> Vector {
        Vector {
            x: self*v.x,
            y: self*v.y
        }
    }
}

// Product of symetric matrix and vector
// ⎡x'⎤ = ⎡κ₁  τ⎤*⎡x⎤
// ⎣y'⎦ = ⎣τ  κ₂⎦ ⎣y⎦
impl<'a, 'b> Mul<&'b Vector> for &'a SymMatrix {
    type Output = Vector;
    fn mul(self, v: &'b Vector) -> Vector {
        Vector {
            x: self.element_11*v.x + self.element_12*v.y,
            y: self.element_12*v.x + self.element_22*v.y
        }
    }
}

// Product of ez-matrix and vector
// ⎡x'⎤ = ⎡η -ζ⎤*⎡x⎤
// ⎣y'⎦ = ⎣ζ  η⎦ ⎣y⎦
impl<'a, 'b> Mul<&'b Vector> for &'a EzMatrix {
    type Output = Vector;
    fn mul(self, v: &'b Vector) -> Vector {
        Vector {
            x: self.element_11*v.x - self.element_21*v.y,
            y: self.element_21*v.x + self.element_11*v.y
        }
    }
}

// Product of symetric matrix and ez-matrix
//     ⎡a11 a12⎤ = ⎡κ₁  τ⎤*⎡η -ζ⎤ = ⎡κ₁*η + τ*ζ  -κ₁*ζ + τ*η⎤
//     ⎣a21 a22⎦   ⎣τ  κ₂⎦ ⎣ζ  η⎦   ⎣τ*η + κ₂*ζ  -τ*ζ + κ₂*η⎦
impl<'a, 'b> Mul<&'b EzMatrix> for &'a SymMatrix {
    type Output = Matrix;
    fn mul(self, other: &'b EzMatrix) -> Matrix {
        let taueta = self.element_12*other.element_11;
        let tauzeta = self.element_12*other.element_21;
        Matrix {
            element_11: self.element_11*other.element_11 + tauzeta,
            element_12: -self.element_11*other.element_21 + taueta,
            element_21: taueta + self.element_22*other.element_21,
            element_22: -tauzeta + self.element_22*other.element_11
        }
    }
}
