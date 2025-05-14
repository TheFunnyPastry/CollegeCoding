#include "date.h"
#include <iostream>
#include <iomanip>
#include <cstring> // For string operations like strlen and strchr

using namespace std;

// Default constructor
Date::Date() : month(1), day(1), year(2025) {}

// Parameterized constructor
Date::Date(int m, int d, int y) {
    if (IsValidDate(m, d, y)) {
        month = m;
        day = d;
        year = y;
    } else {
        month = 1;
        day = 1;
        year = 2025;
    }
}

// Conversion constructor
Date::Date(const char* dateStr) {
    int m = 0, d = 0, y = 0;
    const char* firstSlash = strchr(dateStr, '/');
    const char* secondSlash = strchr(dateStr,'/');

    if (firstSlash && secondSlash) {
        // Parse month
        for (const char* p = dateStr; p < firstSlash; ++p) {
            if (*p >= '0' && *p <= '9') {
                m = m * 10 + (*p - '0');
            } else {
                m = 0; // Invalid character
                break;
            }
        }

        // Parse day
        for (const char* p = firstSlash + 1; p < secondSlash; ++p) {
            if (*p >= '0' && *p <= '9') {
                d = d * 10 + (*p - '0');
            } else {
                d = 0; // Invalid character
                break;
            }
        }

        // Parse year
        for (const char* p = secondSlash + 1; *p != '\0'; ++p) {
            if (*p >= '0' && *p <= '9') {
                y = y * 10 + (*p - '0');
            } else {
                y = 0; // Invalid character
                break;
            }
        }
    }

    // Validate and initialize
    if (IsValidDate(m, d, y)) {
        month = m;
        day = d;
        year = y;
    } else {
        month = 1;
        day = 1;
        year = 2025;
    }
}

// Input function
void Date::Input() {
    int m, d, y;
    char delimiter1, delimiter2;

    while (true) {
        cout << "Input date in month/day/year format: ";
        cin >> m >> delimiter1 >> d >> delimiter2 >> y;

        if (delimiter1 == '/' && delimiter2 == '/' && IsValidDate(m, d, y)) {
            month = m;
            day = d;
            year = y;
            break;
        } else {
            cout << "Invalid date. Try again: ";
        }
    }
}

// Accessor functions
int Date::GetMonth() const { return month; }
int Date::GetDay() const { return day; }
int Date::GetYear() const { return year; }

// Set function
bool Date::Set(int m, int d, int y) {
    if (IsValidDate(m, d, y)) {
        month = m;
        day = d;
        year = y;
        return true;
    }
    return false;
}

// Increment function
void Date::Increment() {
    day++;
    if (day > DaysInMonth(month, year)) {
        day = 1;
        month++;
        if (month > 12) {
            month = 1;
            year++;
        }
    }
}

// Decrement function
void Date::Decrement() {
    if (day > 1) {
        day--;
    } else {
        if (month > 1) {
            month--;
            day = DaysInMonth(month, year);
        } else {
            if (year > 1900) {
                year--;
                month = 12;
                day = 31;
            }
        }
    }
}

// DayofWeek function
int Date::DayofWeek() const {
    static int t[] = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};
    int y = year;
    if (month < 3) y--;
    return (y + y / 4 - y / 100 + y / 400 + t[month - 1] + day) % 7;
}

// Compare function
int Date::Compare(const Date& d) const {
    if (year < d.year || (year == d.year && month < d.month) || (year == d.year && month == d.month && day < d.day)) {
        return -1;
    } else if (year == d.year && month == d.month && day == d.day) {
        return 0;
    } else {
        return 1;
    }
}

// ShowByDay function
void Date::ShowByDay() const {
    static const string daysOfWeek[] = {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"};
    cout << daysOfWeek[DayofWeek()] << " " << month << "/" << day << "/" << year;
}

// ShowByMonth function
void Date::ShowByMonth() const {
    static const string months[] = {"January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"};
    cout << months[month - 1] << "    " << year << endl;
    cout << "Su    Mo    Tu    We    Th    Fr    Sa" << endl;

    int firstDay = DayofWeek() - (day - 1) % 7;
    if (firstDay < 0) firstDay += 7;

    for (int i = 0; i < firstDay; i++) {
        cout << "      ";
    }

    for (int d = 1; d <= DaysInMonth(month, year); d++) {
        cout << setw(2) << setfill('0') << d << "    ";
        if ((firstDay + d) % 7 == 0) cout << endl;
    }
    cout << endl;
}

// Helper functions
bool Date::IsLeapYear(int y) const {
    return (y % 4 == 0 && y % 100 != 0 || y % 400 == 0);
}

int Date::DaysInMonth(int m, int y) const {
    static const int daysInMonth[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (m == 2 && IsLeapYear(y)) return 29;
    return daysInMonth[m - 1];
}

bool Date::IsValidDate(int m, int d, int y) const {
    return y >= 1900 && m >= 1 && m <= 12 && d >= 1 && d <= DaysInMonth(m, y);
}