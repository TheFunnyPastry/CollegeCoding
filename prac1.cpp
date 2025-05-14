class Circle {
    public:
        Circle();		// this is a constructor (known as a "default 
			//    constructor" because no parameters)
        Circle(double r);	// this is also a constructor, with a parameter
        void setCenter( double x, double y);
        void setRadius(int r);
        void Draw()const;
        double getArea()const;    
    private:
        double radius;
        double center_x;
        double cetner_y;
};
class TimeType
{
public:
   void Set(int, int, int);	// set the time ("setter" function)
   void Increment();		// increment the timer by one second
				//  involves changing object's data

   void Display() const;	// output the time -- accessor

   // "getter" functions -- all accessors
   int GetHours() const;
   int GetMinutes() const;
   int GetSeconds() const;

private:
   int hours;
   int minutes;
   int seconds;
};