// Computes the value of cubic spline at given x from evenly spaced grid domain,
// using numerically stable Cox-DeBoor algorithm.
//
// Spline knots are assumed to be i*μ, i = 0, 1, ..., L, where
// μ > 0 is grid spacing. The domain is assumed to be interval [0, μ*L].
// If j is such that j*μ ≤ x ≤ (j + 1)*μ, then the value of cubic spline at x is
//             3          ⎛x - (j+k-3)μ⎞
//     s(x) =  ∑  c[j+k]*B⎜⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎟,
//            k=0         ⎝     μ      ⎠
// where c is the array of spline coefficients and B is the basis cubic spline over
// knots (0, 1, 2, 3, 4). The number of spline coefficients (i.e. the length
// of array c) is L + 3.
// Computes s'(x) too!
//
// Arguments:
// x ... value, at which s(x) and s'(x) should be computed
// coefficients ... array of spline coefficients c
// grid_spacing ... grid spacing μ
//
// Returns:
// pair (s(x), s'(x))
//
pub fn cubic_interpolate(x: f64, coefficients: &[f64], grid_spacing: f64) -> (f64, f64) {
    // index j, such that j*μ ≤ x ≤ (j + 1)*μ
    // TODO: preveri, če je x sploh v območju.
    let j = (x/grid_spacing) as usize;
    // x = j*μ + r
    let r = x%grid_spacing;
    // temporary array of relavant coefficients, used for recursive Cox-DeBoor algorithm
    let mut c = [0.0; 4];
    c.clone_from_slice(&coefficients[j .. j + 4]);

    let mut d = [c[1] - c[0], c[2] - c[1], c[3] - c[2]];
    let p = r/grid_spacing; // p = r/μ

    // Calculating value s(x):
    // coefficients c^[1](x)
    c[0] = (2.0*c[1] + c[0] + p*d[0])/3.0;
    c[1] = (c[2] + 2.0*c[1] + p*d[1])/3.0;
    c[2] = c[2] + p*d[2]/3.0;
    // coefficients c^[2](x)
    c[0] = (c[1] + c[0] + p*(c[1] - c[0]))/2.0;
    c[1] = c[1] + p*(c[2] - c[1])/2.0;
    // coefficient c^[3](x) = s(x)
    c[0] = c[0] + p*(c[1] - c[0]);

    // Calculating derivative s'(x):
    // coefficients d^[1](x)
    d[0] = (d[1] + d[0] + p*(d[1] - d[0]))/2.0/grid_spacing;
    d[1] = (d[1] + p*(d[2] - d[1])/2.0)/grid_spacing;
    // coefficient d^[2](x) = s'(x)
    d[0] = d[0] + p*(d[1] - d[0]);

    (c[0], d[0]) // return result
}

fn cox_de_boor_3(c: &mut [f64; 4], p: f64) -> f64 {
    // Calculating value s³(x):
    // coefficients c^[1](x)
    c[0] = (2.0*c[1] + c[0] + p*(c[1] - c[0]))/3.0;
    c[1] = (c[2] + 2.0*c[1] + p*(c[2] - c[1]))/3.0;
    c[2] = c[2] + p*(c[3] - c[2])/3.0;
    // coefficients c^[2](x)
    c[0] = (c[1] + c[0] + p*(c[1] - c[0]))/2.0;
    c[1] = c[1] + p*(c[2] - c[1])/2.0;
    // coefficient c^[3](x) = s(x)
    c[0] + p*(c[1] - c[0])
}

fn cox_de_boor_2(c: &mut [f64; 3], p: f64) -> f64 {
    // Calculating value s²(x):
    // coefficients c^[1](x)
    c[0] = (c[1] + c[0] + p*(c[1] - c[0]))/2.0;
    c[1] = c[1] + p*(c[2] - c[1])/2.0;
    // coefficient c^[2](x) = s(x)
    c[0] + p*(c[1] - c[0])
}

fn cox_de_boor_1(c: &mut [f64; 2], p: f64) -> f64 {
    // Calculating value s¹(x):
    // coefficient c^[1](x) = s(x)
    c[0] + p*(c[1] - c[0])
}

use gridfunction::Grid;

pub fn cubic_interpolate_2d(x: f64, y: f64, coefficients: &[f64], grid: Grid) -> f64 {
    let (jx, px) = {
        let z = x / grid.spacing_x;
        (z as usize, z - z.floor())
    };
    let (jy, py) = {
        let z = y / grid.spacing_y;
        (z as usize, z - z.floor())
    };
    let dima = grid.size_x + 2; // the size of first dimension of 2D array of bicubic spline coefficients
    let j = jy*dima + jx;
    let js = [j, j + dima, j + 2*dima, j + 3*dima];
    let mut c_inter = [0.0; 4];
    for i in 0..4 {
        let mut c = [0.0; 4];
        c.clone_from_slice(&coefficients[js[i] .. js[i] + 4]);
        c_inter[i] = cox_de_boor_3(&mut c, px);
    }
    cox_de_boor_3(&mut c_inter, py)
}

pub fn cubic_interpolate_with_derivatives_2d(x: f64, y: f64, coefficients: &[f64], grid: Grid) -> (f64, f64, f64) {
    // TODO: Kaj pa, če (x,y) ni na območju, ki ga zajema grid??
    let (jx, px) = {
        let z = x / grid.spacing_x;
        (z as usize, z - z.floor())
    };
    let (jy, py) = {
        let z = y / grid.spacing_y;
        (z as usize, z - z.floor())
    };
    let dima = grid.size_x + 2; // the size of first dimension of 2D array of bicubic spline coefficients
    let j = jy*dima + jx;
    let js = [j, j + dima, j + 2*dima, j + 3*dima];
    let mut c_inter = [0.0; 4]; // here will be values of interpolation along x axis
    let mut d_inter = [0.0; 4]; // here will be derivatives of interpolation along x axis
    for i in 0..4 {
        let mut c = [0.0; 4];
        c.clone_from_slice(&coefficients[js[i] .. js[i] + 4]);
        let mut d = [(c[1] - c[0])/grid.spacing_x,
                     (c[2] - c[1])/grid.spacing_x,
                     (c[3] - c[2])/grid.spacing_x];
        c_inter[i] = cox_de_boor_3(&mut c, px);
        d_inter[i] = cox_de_boor_2(&mut d, px);
    }
    let mut dy = [(c_inter[1] - c_inter[0])/grid.spacing_y,
                  (c_inter[2] - c_inter[1])/grid.spacing_y,
                  (c_inter[3] - c_inter[2])/grid.spacing_y];
    // value s(x,y) and derivatives ∂s(x,y)/∂x and ∂s(x,y)/∂y
    (cox_de_boor_3(&mut c_inter, py), cox_de_boor_3(&mut d_inter, py), cox_de_boor_2(&mut dy, py))
}