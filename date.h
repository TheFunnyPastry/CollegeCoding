#define DATE_H

class Date
{
public:
    // constructors
    Date();                             // default
    Date(int month, int day, int year); // parameterized
    Date(const char *strDate);          // conversion

    // member functions
    void Input();
    int GetMonth() const;
    int GetDay() const;
    int GetYear() const;
    bool Set(int m, int d, int y);
    void Increment();   
    void Decrement();
    int DayofWeek() const;
    int Compare(const Date &d) const;
    void ShowByDay() const;
    void ShowByMonth() const;

private:
    // date values
    int month;
    int day;
    int year;
    // helper functions
    bool IsLeapYear(int year) const;
    int DaysInMonth(int month, int year) const;
    bool IsValidDate(int m, int d, int y) const;
};