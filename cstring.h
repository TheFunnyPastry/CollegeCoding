#include<iostream>

s1 = s2+s3;

class Bstring{
    friend ostream& operator<< (ostream& os, const BString& s);
    friend istream& operator>> (istream& is, BString& s);

    friend Bstring operator+(const);
public:
    
    int GetSize() const;
    Bstring Substring(int index, int len);
private:
    char * str;
    int size;
}