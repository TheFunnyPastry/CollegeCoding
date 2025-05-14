#include <iostream>
#include "Stack.h"
#include "evaluateRPN.h"

int main() {
    // --- Test Stack functionality ---
    std::cout << "Testing Stack:" << std::endl;
    Stack<int> s;
    std::cout << "Pushing elements: 10, 20, 30" << std::endl;
    s.push(10);
    s.push(20);
    s.push(30);
    
    std::cout << "Current top: " << s.top() << std::endl;
    
    std::cout << "Popped elements: ";
    while (!s.empty()) {
        std::cout << s.pop() << " ";
    }
    std::cout << std::endl;
    
    // --- Test evaluateRPN functionality ---
    std::cout << "Testing evaluateRPN:" << std::endl;
    std::string expr = "3 4 + 2 * 1 +";
    int result = evaluateRPN<int>(expr);
    std::cout << "RPN Expression: \"" << expr << "\" evaluates to " << result << std::endl;
    
    return 0;
}
