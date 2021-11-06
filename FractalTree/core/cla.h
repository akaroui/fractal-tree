#ifndef GEODESIC_H
#define GEODESIC_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <sstream>


namespace geodesic
{

template<typename T> using Vec=std::vector<T>;

typedef float Real;
typedef Vec<Real> VReal;
typedef Vec<VReal> VVReal;

//typedef unsigned int Int;
typedef int Int;
typedef Vec<Int> VInt;
typedef Vec<VInt> VVInt;

typedef Vec<bool> VBool;
typedef std::string String;
typedef Vec<String> VString;



template <class... Ts>
void Print(Ts&&... args){
	std::stringstream o;  // o << "";
	((o << args << ' '), ...);
	std::cout << o.str() << std::endl;
}


template<typename T>
bool isIn(T e, Vec<T> vec)
{
	for(T elem : vec) if(elem==e) return true;
	return false;
}


template<class T>
T vmin(std::vector<T> const &v) {
	typename std::vector<T>::const_iterator result = std::min_element(v.begin(), v.end());
    return v[std::distance(v.begin(), result)];
}


template<class T>
T norm(std::vector<T> const &a, std::vector<T> const &b) {
	std::vector<T> r = {b[0]-a[0], b[1]-a[1], b[2]-a[2]};
	return std::sqrt(std::inner_product(r.begin(), r.end(), r.begin(), 0));
}


Real c_geodesic_distance(Int a,
						 Int b,
						 Int n,
						 VVReal const &pts,
						 VVInt const &pap,
						 VReal const &cond) {
	if (a == b) return 0;

    VReal d(n);
    std::fill(d.begin(), d.end(), 0);  // distances

    VBool visited(n);
    std::fill(visited.begin(), visited.end(), false);
    visited[0] = true;
    visited[a] = true;

    Real dmin, dcur, nc;
    Int n_iter(0);

    VReal endnodes;
    VInt origins, new_origins, around;
    origins.push_back(a);

	while (origins.size() > 0 and n_iter < n) {
		n_iter++;
		dmin = (endnodes.size() > 0) ? vmin(endnodes) : 1e10;
		new_origins.clear();

		for (Int y : origins) {
			around.clear();
			for (int x : pap[y]) {
				if (x != -1) around.push_back(x);
			}
			for (Int x : around) {
				nc = std::max(0.5 * (cond[y] + cond[x]), 1e-6);  // conductivity threshold
				dcur = d[y] + norm(pts[x], pts[y]) / nc;
				if (x == b) endnodes.push_back(dcur);
				else if (!visited[x]) {
					if (dcur < dmin) new_origins.push_back(x);
					d[x] = dcur;
					visited[x] = true;
				}
				else d[x] = std::min(dcur, d[x]);
			}
		}
		origins = new_origins;
	}

	if (endnodes.size() == 0) return -1;
	return vmin(endnodes);
}


}  // namespace geodesic

#endif