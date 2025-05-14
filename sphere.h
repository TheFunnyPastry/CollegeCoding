#define sphere_h
#include <string>

class Sphere{
private:
    //sphere values
    double radius;
    char color;
    //helper function
    bool validColor(char c)const;
    bool randColor();
    std::string colortostr()const;
    
    public:
    //constructors
    Sphere();
    Sphere(double r, char c = '\0');
    //getter functions
    double getRad() const;
    char getColor() const;
    double getDiameter() const;
    double getSurfaceArea() const;
    double getVolume()const;
    //setter functions
    void SetRadius(double r);
    void SetColor(double r);
    void grow(double amount);
    void shrink(double amount);
    void randomizeColor();
    void printSummary(int precision = 2);

};