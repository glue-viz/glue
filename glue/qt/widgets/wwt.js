function PointCollection(wwt, x, y) {

    var result = {};
    result.x = x;
    result.y = y;
    result.cricles = [];
    
    result.setup = function{
        this.clear();
        for(i = 0; i < this.x.length; i+=1) {
            this.circles.push(this.wwt.createCircle());
        }
        this.show();
    };
    
    result.clear = function() {
        for(c in this.circles) {
            this.wwt.removeAnnotation(c);
            }
        this.circles = [];
        };   
    
    result.updateData = function(x, y) {
        this.x = x;
        this.y = y;
        this.setup();
    };
    
    result.show = function() {
        for(c in this.circles) {
            this.wwt.addAnnotation(c);
        }        
    };
    
    result.hide = function() {
        for(c in this.circles) {
            this.wwt.removeAnnotation(c);
        }           
    };
    
    result.setColor = function(color) {
        for(c in this.circles) {
            c.set_fill(true);
            c.set_fillColor(color);
        }
    };
    result.setup();
    return result;
}