#include <string>
#include <map>

namespace cvl{

/**
 * @brief The Sym struct
 * - provides multiplication, division
 */
struct Sym {
    std::map<std::string, int> comps;
    Sym()=default;
    Sym(std::string n, int exponent=1);


    // display string
    std::string str() const;
    // sortable, unique permutation, invariant hash
    std::string hash() const;
    // removes a^0s
    void simplify();

    // supported operations:
    Sym& operator*=(Sym a);
    Sym& operator/=(Sym a);
};

Sym operator*(Sym a, Sym b);
Sym operator/(Sym a, Sym b);
bool operator<(Sym a, Sym b);



struct Symb{
    std::map<Sym,double> koeffs;
    Symb()=default;
    Symb(double d);
    Symb(Sym a, double k=1.0);
    void clear_zeros();
    std::string str();
    Symb& operator+=(Symb b);
    Symb& operator*=(Symb b);
};
Symb operator+(Symb a, Symb b);
Symb operator*(Symb as, Symb bs);
Symb operator*(Symb as, double d);
bool operator==(Symb s, double d);
bool operator<(Symb a, Symb b);
Symb operator-(Symb s);

std::ostream& operator<<(std::ostream& os, Sym s);
std::ostream& operator<<(std::ostream& os, Symb s);
} // end namespace cvl




