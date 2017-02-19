struct Grid {
    size_x: i32,
    size_y: i32,
    spacing_x: f64,
    spacing_y: f64
}

struct GridImage {
    grid: Grid,
    values: Vec<f64>
}

impl GridImage {

/*
    fn create(size_x: i32, size_y: i32, spacing_x: f64, spacing_y: f64, values: Vec<f64>) -> Result<GridImage, String> {
        if size_x < 1 || size_y < 1 || spacing_x <= 0.0 || spacing_y <= 0.0 {
            Err("Invalid arguments for Grid in constructor for GridImage.")
        }
        // TODO: Nedokončano
    }
*/

    fn interpolate(&self, x: f64, y: f64) -> f64 {
        let x_ind_low = (x / self.grid.size_x) as i32;
        let y_ind_low = (y / self.grid.size_y) as i32;
        let index_low = (y_ind_low*self.grid.size_x + x_ind_low) as usize;
        let a = (x_ind_low as f64)*self.grid.size_x;
        let b = a + self.grid.size_x;
        let c = (y_ind_low as f64)*self.grid.size_y;
        let d = c + self.grid.size_y;
        let dx_low = x - a;
        let dx_high = b - x;
        let dy_low = y - c;
        let dy_high = d - y;
        dx_high*dy_high*self.values[index_low] +
            dx_low*dy_high*self.values[index_low + 1usize] +
            dx_high*dy_low*self.values[index_low + self.grid.size_x] +
            dx_low*dy_low*self.values[index_low + self.grid.size_x + 1usize]
        // TODO: Interpolacija vrednosti izven območja
        // FIXME: Deli z dolžino intervalov. Zaenkrat velja samo, če so dolgi 1.
    }

}

