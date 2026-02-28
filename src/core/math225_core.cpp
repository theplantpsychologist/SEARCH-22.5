#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <numeric>
#include <cmath>
#include <string>
#include <stdexcept>
#include <functional>
#include <map>
#include <algorithm>

namespace py = pybind11;
// Global cache for Python objects to avoid re-importing/re-creating
// struct MathGlobals {
//     py::object gcd;
//     py::int_ zero;
//     py::int_ one;

//     MathGlobals() {
//         py::object math = py::module_::import("math");
//         gcd = math.attr("gcd");
//         zero = py::int_(0);
//         one = py::int_(1);
//     }
// };
// static MathGlobals* g_math = nullptr;


// --- 1. Fast C++ Fraction ---
struct Fraction {
    int64_t num, den; // Use Python's BigInts

    // The constructor now enforces strict normalization
    Fraction(int64_t n = 0, int64_t d = 1) {
        if (d == 0) throw std::invalid_argument("Denominator is zero");
        if (d < 0) { n = -n; d = -d; }
        
        // Immediate simplification keeps numbers within 64-bit limits
        int64_t g = std::gcd(std::abs(n), std::abs(d));
        num = n / g;
        den = d / g;
    }

    // Boilerplate for operators using Python's logic
    Fraction operator+(const Fraction& o) const {
        return Fraction(num * o.den + o.num * den, den * o.den);
    }
    Fraction operator-(const Fraction& o) const {
        return Fraction(num * o.den - o.num * den, den * o.den);
    }
    Fraction operator*(const Fraction& o) const {
        return Fraction(num * o.num, den * o.den);
    }
    Fraction operator/(const Fraction& o) const {
        return Fraction(num * o.den, den * o.num);
    }

   bool operator==(const Fraction& o) const {
        return num == o.num && den == o.den;
    }

    bool operator<(const Fraction& o) const {
        // Use long double for overflow-safe comparison
        return (long double)num * o.den < (long double)o.num * den;
    } 
    
    bool operator>(const Fraction& o) const {
        return (long double)num * o.den > (long double)o.num * den;
    }
    
    Fraction operator-() const {
        return Fraction(-num, den);
    }
    
    double to_float() const {
        return (double)num / (double)den;
    }
};

// --- 2. C++ AplusBsqrt2 ---
struct AplusBsqrt2 {
    Fraction A, B;
    AplusBsqrt2(Fraction a, Fraction b) : A(a), B(b) {}
    AplusBsqrt2(const Fraction& a) : A(a), B(Fraction(0)) {}
    AplusBsqrt2(int64_t a) : A(Fraction(a)), B(Fraction(0)) {}

    AplusBsqrt2 operator+(const AplusBsqrt2& o) const { return AplusBsqrt2(A + o.A, B + o.B); }
    AplusBsqrt2 operator-(const AplusBsqrt2& o) const { return AplusBsqrt2(A - o.A, B - o.B); }
    
    AplusBsqrt2 operator*(const AplusBsqrt2& o) const {
        Fraction two(2, 1);
        return AplusBsqrt2(A * o.A + two * B * o.B, A * o.B + B * o.A);
    }

    AplusBsqrt2 operator/(const AplusBsqrt2& o) const {
        Fraction two(2, 1);
        Fraction denom = o.A * o.A - two * o.B * o.B;
        if (denom.num == 0) throw std::invalid_argument("Division by zero in AplusBsqrt2");
        Fraction newA = (A * o.A - two * B * o.B) / denom;
        Fraction newB = (B * o.A - A * o.B) / denom;
        return AplusBsqrt2(newA, newB);
    }
    
    bool operator<(const AplusBsqrt2& o) const { return this->to_float() < o.to_float(); }
    bool operator>(const AplusBsqrt2& o) const { return this->to_float() > o.to_float(); }
    
    // Safety check relies on the new Overflow-Proof Fraction equality
    bool operator==(const AplusBsqrt2& o) const { return A == o.A && B == o.B; }
    
    AplusBsqrt2 operator-() const { return AplusBsqrt2(-A, -B); }
    double to_float() const { return A.to_float() + B.to_float() * 1.4142135623730951; }
    
    int sign() const {
        double val = to_float();
        return (val > 0) - (val < 0);
    }
};
// Helper to create Fraction from int
inline Fraction F(int n) { return Fraction(n, 1); }
inline Fraction F(int n, int d) { return Fraction(n, d); }

// --- 3. C++ Vertex4D ---
struct Vertex4D {
    Fraction x, y, z, w;
    
    // Default constructor
    Vertex4D() : x(Fraction(0)), y(Fraction(0)), z(Fraction(0)), w(Fraction(0)) {}
    Vertex4D(Fraction _x, Fraction _y, Fraction _z, Fraction _w) : x(_x), y(_y), z(_z), w(_w) {}

    bool operator==(const Vertex4D& o) const { return x == o.x && y == o.y && z == o.z && w == o.w; }
    
    // Lexicographic comparison for sorting
    bool operator<(const Vertex4D& o) const {
        if (!(x == o.x)) return x < o.x;
        if (!(y == o.y)) return y < o.y;
        if (!(z == o.z)) return z < o.z;
        return w < o.w;
    }
    
    Vertex4D operator+(const Vertex4D& o) const { return Vertex4D(x + o.x, y + o.y, z + o.z, w + o.w); }
    Vertex4D operator-(const Vertex4D& o) const { return Vertex4D(x - o.x, y - o.y, z - o.z, w - o.w); }
    Vertex4D operator*(const Fraction& o) const { 
        return Vertex4D(x * o, y * o, z * o, w * o); 
    }
    
    Vertex4D operator*(const AplusBsqrt2& o) const {
        // Result = V * o.A + (V * o.B) * SQRT2
        // We inline the SQRT2 matrix-vector multiplication for speed:
        // (V * o.B) * SQRT2 results in:
        //   x_part = o.B * (y - w)
        //   y_part = o.B * (x + z)
        //   z_part = o.B * (y + w)
        //   w_part = o.B * (z - x)
        
        return Vertex4D(
            x * o.A + o.B * (y - w),
            y * o.A + o.B * (x + z),
            z * o.A + o.B * (y + w),
            w * o.A + o.B * (z - x)
        );
    }

    std::vector<double> to_cartesian() const {
        return {
            x.to_float() + y.to_float() * sqrt(0.5) - w.to_float() * sqrt(0.5),
            z.to_float() + y.to_float() * sqrt(0.5) + w.to_float() * sqrt(0.5)
        };
    }

    AplusBsqrt2 dot_product(const Vertex4D& o) const {
        Fraction sx1 = y - w, sx2 = x, sy1 = y + w, sy2 = z;
        Fraction ox1 = o.y - o.w, ox2 = o.x, oy1 = o.y + o.w, oy2 = o.z;
        Fraction two = F(2);
        return AplusBsqrt2(
            (sx1 * ox1 + two * sx2 * ox2) + (sy1 * oy1 + two * sy2 * oy2),
            (sx1 * ox2 + sx2 * ox1) + (sy1 * oy2 + sy2 * oy1)
        );
    }

    std::optional<int> angle_to(const Vertex4D& other) const {
        if (*this == other) return std::nullopt;
        Vertex4D d = other - *this;
        Fraction dx_A = d.y - d.w, dx_B = d.x, dy_A = d.y + d.w, dy_B = d.z;

        auto sign = [](const Fraction& a, const Fraction& b) {
            double val = a.to_float() + b.to_float() * 1.4142135623730951;
            return (val > 0) ? 1 : -1;
        };

        if (dx_A.num == 0 && dx_B.num == 0) return (sign(dy_A, dy_B) > 0) ? 4 : 12;
        if (dy_A.num == 0 && dy_B.num == 0) return (sign(dx_A, dx_B) > 0) ? 0 : 8;

        Fraction two = F(2);
        int s = (sign(dy_A, dy_B) > 0) ? 0 : 8;
        if (dy_A == -dx_A + two * dx_B && dy_B == dx_A - dx_B) return 1 + s;
        if (dy_A == dx_A && dy_B == dx_B) return 2 + s;
        if (dy_A == dx_A + two * dx_B && dy_B == dx_A + dx_B) return 3 + s;
        if (dy_A == -dx_A - two * dx_B && dy_B == -dx_A - dx_B) return 5 + s;
        if (dy_A == -dx_A && dy_B == -dx_B) return 6 + s;
        if (dy_A == dx_A - two * dx_B && dy_B == -dx_A + dx_B) return 7 + s;
        return std::nullopt;
    }
};

// --- 4. Geometric helpers ---
Vertex4D apply_reflection(int angle, const Fraction& x, const Fraction& y, const Fraction& z, const Fraction& w) {
    switch (angle % 8) {
        case 0: return Vertex4D(x, -w, -z, -y);
        case 1: return Vertex4D(y, x, -w, -z);
        case 2: return Vertex4D(z, y, x, -w);
        case 3: return Vertex4D(w, z, y, x);
        case 4: return Vertex4D(-x, w, z, y);
        case 5: return Vertex4D(-y, -x, w, z);
        case 6: return Vertex4D(-z, -y, -x, w);
        case 7: return Vertex4D(-w, -z, -y, -x);
        default: return Vertex4D(F(0), F(0), F(0), F(0));
    }
}

Vertex4D reflect(const Vertex4D& v1, const Vertex4D& v2, const Vertex4D& p) {
    auto angle = v1.angle_to(v2);
    if (!angle) throw std::runtime_error("Undefined reflection angle");
    Vertex4D relative = p - v1;
    Vertex4D reflected = apply_reflection(*angle, relative.x, relative.y, relative.z, relative.w);
    return reflected + v1;
}


Vertex4D apply_symmetry(const Vertex4D& v, int rot_index, bool mirror) {
    Vertex4D p;

    // Direct mapping for rotations (0, 2, 4, 6, 8, 10, 12, 14)
    // rot_index is in 22.5 units, so rot_index/2 is the number of 45-degree steps
    switch (rot_index) {
        case 0:  p = v; break;
        case 2:  p = Vertex4D(-v.w,  v.x,  v.y,  v.z); break;
        case 4:  p = Vertex4D(-v.z, -v.w,  v.x,  v.y); break;
        case 6:  p = Vertex4D(-v.y, -v.z, -v.w,  v.x); break;
        case 8:  p = Vertex4D(-v.x, -v.y, -v.z, -v.w); break;
        case 10: p = Vertex4D( v.w, -v.x, -v.y, -v.z); break;
        case 12: p = Vertex4D( v.z,  v.w, -v.x, -v.y); break;
        case 14: p = Vertex4D( v.y,  v.z,  v.w, -v.x); break;
        // default: p = v; // Should not happen given your constraints
    }

    if (mirror) {
        // Optimization: Create temp variables to avoid overwriting values mid-swap
        Fraction mx = p.x;
        Fraction my = -p.w;
        Fraction mz = -p.z;
        Fraction mw = -p.y;
        p = Vertex4D(mx, my, mz, mw);
    }
    return p;
}

py::tuple internal_freeze(const std::vector<Vertex4D>& verts, 
                          const std::vector<std::pair<int, int>>& edges,
                          const std::vector<std::vector<int>>& faces,
                          const std::vector<std::vector<std::vector<std::pair<int, int>>>>& instances) {
    py::list k;
    k.append(verts.size()); k.append(edges.size()); k.append(faces.size());
    for(const auto& v : verts) {
        k.append(v.x.num); k.append(v.x.den); k.append(v.y.num); k.append(v.y.den);
        k.append(v.z.num); k.append(v.z.den); k.append(v.w.num); k.append(v.w.den);
    }
    for(const auto& e : edges) { k.append(e.first); k.append(e.second); }
    for(const auto& f : faces) {
        k.append(f.size());
        for(int e_idx : f) k.append(e_idx);
    }
    for(const auto& stack : instances) {
        k.append(stack.size());
        for(const auto& inst : stack) {
            k.append(inst.size());
            for(const auto& c : inst) { k.append(c.first); k.append(c.second); }
        }
    }
    return py::tuple(k);
}

std::vector<int> canonicalize_face(std::vector<int> face) {
    if (face.empty()) return face;
    auto min_it = std::min_element(face.begin(), face.end());
    std::rotate(face.begin(), min_it, face.end());
    return face;
}

struct FaceUnit {
    std::vector<int> edge_indices;
    std::vector<std::vector<std::pair<int, int>>> instance_stack;
    int original_face_idx;
};

// Returns {canonical_edges, direction_was_flipped, offset_used}
std::tuple<std::vector<int>, bool, int> canonicalize_face_indices(std::vector<int> edges) {
    if (edges.empty()) return std::make_tuple(edges, false, 0);
    
    std::vector<int> best = edges;
    bool best_flipped = false;
    int best_offset = 0;

    int n = (int)edges.size();
    bool flips[] = {false, true};
    for (int fi = 0; fi < 2; ++fi) {
        bool flip = flips[fi];
        std::vector<int> current = edges;
        if (flip) std::reverse(current.begin(), current.end());
        
        for (int rot = 0; rot < n; ++rot) {
            std::rotate(current.begin(), current.begin() + 1, current.end());
            if (current < best) {
                best = current;
                best_flipped = flip;
                best_offset = (rot + 1) % n;
            }
        }
    }
    return std::make_tuple(best, best_flipped, best_offset);
}

// A structure to represent the full state of an instance for sorting
struct InstanceRef {
    int face_idx;
    int inst_idx;
    std::vector<long long> signature;

    bool operator<(const InstanceRef& other) const {
        return signature < other.signature;
    }
};

// Define a struct to hold the C++ representation for comparison
struct State {
    std::vector<Vertex4D> v;
    std::vector<std::pair<int, int>> e;
    std::vector<std::vector<int>> f;
    std::vector<std::vector<std::vector<std::pair<int, int>>>> i;

    // Use C++ operator< for fast state comparison
    bool operator<(const State& o) const {
        if (v != o.v) return v < o.v;
        if (e != o.e) return e < o.e;
        if (f != o.f) return f < o.f;
        return i < o.i;
    }
};

py::tuple canonicalize_cpp(
    const std::vector<Vertex4D>& vertices, 
    const std::vector<std::pair<int, int>>& edges,
    const std::vector<std::vector<int>>& faces,
    const std::vector<std::vector<std::vector<std::pair<int, int>>>>& instances) 
{
    py::object best_key = py::none(); // Start as None
    std::unique_ptr<State> best_state_ptr = nullptr;
    bool mirrors[] = {false, true};
    for (int mi = 0; mi < 2; ++mi) {
        bool mirror = mirrors[mi];
        for (int rot = 0; rot < 16; rot += 2) {
            // --- 1. VERTEX NORMALIZATION ---
            std::vector<std::pair<Vertex4D, int>> indexed_verts;
            for (size_t i = 0; i < vertices.size(); ++i) {
                Vertex4D v = apply_symmetry(vertices[i], rot, mirror);
                indexed_verts.push_back(std::make_pair(v, (int)i));
            }
            Vertex4D min_v = indexed_verts[0].first;
            for(auto& p : indexed_verts) if(p.first < min_v) min_v = p.first;
            for(auto& p : indexed_verts) p.first = p.first - min_v;
            std::sort(indexed_verts.begin(), indexed_verts.end());

            std::vector<int> v_old_to_new(vertices.size());
            std::vector<Vertex4D> sorted_verts;
            for (size_t i = 0; i < indexed_verts.size(); ++i) {
                sorted_verts.push_back(indexed_verts[i].first);
                v_old_to_new[indexed_verts[i].second] = (int)i;
            }

            // --- 2. EDGE MAPPING ---
            std::vector<std::pair<std::pair<int, int>, int>> indexed_edges;
            for (size_t i = 0; i < edges.size(); ++i) {
                int u = v_old_to_new[edges[i].first];
                int v = v_old_to_new[edges[i].second];
                indexed_edges.push_back(std::make_pair(std::make_pair(std::min(u, v), std::max(u, v)), (int)i));
            }
            std::sort(indexed_edges.begin(), indexed_edges.end());

            std::vector<int> e_old_to_new(edges.size());
            std::vector<std::pair<int, int>> sorted_edges;
            for (size_t i = 0; i < indexed_edges.size(); ++i) {
                sorted_edges.push_back(indexed_edges[i].first);
                e_old_to_new[indexed_edges[i].second] = (int)i;
            }

            // --- 3. FACE UNIT PREPARATION ---
            std::vector<FaceUnit> units;
            for (size_t f = 0; f < faces.size(); ++f) {
                std::vector<int> remapped_e;
                for (int e_idx : faces[f]) remapped_e.push_back(e_old_to_new[e_idx]);
                
                auto result = canonicalize_face_indices(remapped_e);
                std::vector<int> canon_e = std::get<0>(result);
                bool flipped = std::get<1>(result);
                int offset = std::get<2>(result);
                
                std::vector<std::vector<std::pair<int, int>>> canon_stack = instances[f];
                for (auto& inst : canon_stack) {
                    if (flipped) std::reverse(inst.begin(), inst.end());
                    std::rotate(inst.begin(), inst.begin() + offset, inst.end());
                }
                FaceUnit unit;
                unit.edge_indices = canon_e;
                unit.instance_stack = canon_stack;
                unit.original_face_idx = (int)f;
                units.push_back(unit);
            }

            // --- 4. SORT FACES ---
            std::sort(units.begin(), units.end(), [](const FaceUnit& a, const FaceUnit& b) {
                return a.edge_indices < b.edge_indices;
            });

            std::vector<int> f_old_to_new(faces.size());
            std::vector<std::vector<int>> final_faces;
            for (size_t i = 0; i < units.size(); ++i) {
                f_old_to_new[units[i].original_face_idx] = (int)i;
                final_faces.push_back(units[i].edge_indices);
            }

            // --- 5. WEISFEILER-LEHMAN REFINEMENT ---
            std::vector<std::vector<std::vector<long long>>> sigs(units.size());
            for(size_t f=0; f < units.size(); ++f) {
                sigs[f].resize(units[f].instance_stack.size());
                for(size_t i=0; i < units[f].instance_stack.size(); ++i) {
                    for(const auto& conn : units[f].instance_stack[i]) {
                        sigs[f][i].push_back(conn.first == -1 ? -1 : f_old_to_new[conn.first]);
                    }
                }
            }
            
            //More iterations here make it less likely to have ties, but more expensive
            for (int iter = 0; iter < 4; ++iter) {
                auto next_sigs = sigs;
                for (size_t f = 0; f < units.size(); ++f) {
                    for (size_t i = 0; i < units[f].instance_stack.size(); ++i) {
                        for (const auto& conn : units[f].instance_stack[i]) {
                            if (conn.first != -1) {
                                int target_f = f_old_to_new[conn.first];
                                const auto& n_sig = sigs[target_f][conn.second];
                                next_sigs[f][i].insert(next_sigs[f][i].end(), n_sig.begin(), n_sig.end());
                            } else {
                                next_sigs[f][i].push_back(-1);
                            }
                        }
                    }
                }
                sigs = next_sigs;
            }

            // --- 6. BLOCK GROUPING FOR TIES ---
            std::vector<std::vector<int>> current_mapping(units.size());
            struct MutBlock { int f; int start; int size; };
            std::vector<MutBlock> mut_blocks;

            for (size_t f = 0; f < units.size(); ++f) {
                std::vector<int> p(units[f].instance_stack.size());
                std::iota(p.begin(), p.end(), 0);

                // Sort instances by their WL signature
                std::sort(p.begin(), p.end(), [&](int a, int b) {
                    return sigs[f][a] < sigs[f][b];
                });
                current_mapping[f] = p;

                // Identify blocks of instances with IDENTICAL signatures
                int n = p.size();
                int i = 0;
                while (i < n) {
                    int j = i + 1;
                    while (j < n && sigs[f][p[i]] == sigs[f][p[j]]) j++;
                    if (j - i > 1) {
                        // Sort the identical block by original index to ensure we 
                        // always start generating permutations from the same base state
                        std::sort(current_mapping[f].begin() + i, current_mapping[f].begin() + j);
                        MutBlock mb;
                        mb.f = (int)f;
                        mb.start = i;
                        mb.size = j - i;
                        mut_blocks.push_back(mb);
                    }
                    i = j;
                }
            }

           
            // --- 7. OPTIMIZED TIE-BREAKER (Greedy Stabilized Sort) ---
            std::vector<long long> best_inst_key;
            std::vector<std::vector<std::vector<std::pair<int, int>>>> best_final_instances;

            // Helper to build and serialize the layout (Kept from your version)
            std::function<void()> evaluate = [&]() {
                std::map<std::pair<int, int>, int> relabel_map;
                for(size_t f=0; f<units.size(); ++f) {
                    int orig_f = units[f].original_face_idx;
                    for(size_t new_i=0; new_i<current_mapping[f].size(); ++new_i) {
                        int old_i = current_mapping[f][new_i];
                        relabel_map[{orig_f, old_i}] = (int)new_i;
                    }
                }

                std::vector<std::vector<std::vector<std::pair<int, int>>>> test_insts(units.size());
                std::vector<long long> key;

                for(size_t f=0; f<units.size(); ++f) {
                    test_insts[f].resize(current_mapping[f].size());
                    key.push_back((long long)current_mapping[f].size());
                    
                    for(size_t new_i=0; new_i<current_mapping[f].size(); ++new_i) {
                        int old_i = current_mapping[f][new_i];
                        auto inst = units[f].instance_stack[old_i];
                        key.push_back((long long)inst.size());
                        
                        for(auto& conn : inst) {
                            if(conn.first != -1) {
                                int c_orig_f = conn.first;
                                int c_old_i = conn.second;
                                conn.first = f_old_to_new[c_orig_f];
                                conn.second = relabel_map[{c_orig_f, c_old_i}];
                            }
                            key.push_back((long long)conn.first);
                            key.push_back((long long)conn.second);
                        }
                        test_insts[f][new_i] = inst;
                    }
                }

                if(best_inst_key.empty() || key < best_inst_key) {
                    best_inst_key = key;
                    best_final_instances = test_insts;
                }
            };

            // Instead of recursive permutations, sort the tie-blocks using neighbor signatures.
            // This provides a unique canonical ordering in O(N log N) time.
            for (auto& mb : mut_blocks) {
                auto begin_it = current_mapping[mb.f].begin() + mb.start;
                auto end_it = begin_it + mb.size;
                
                std::sort(begin_it, end_it, [&](int a, int b) {
                    // Build a deterministic signature based on neighbor face indices.
                    std::vector<int> sig_a, sig_b;
                    for(const auto& conn : units[mb.f].instance_stack[a]) {
                        sig_a.push_back(conn.first == -1 ? -1 : f_old_to_new[conn.first]);
                    }
                    for(const auto& conn : units[mb.f].instance_stack[b]) {
                        sig_b.push_back(conn.first == -1 ? -1 : f_old_to_new[conn.first]);
                    }
                    return sig_a < sig_b;
                });
            }

            // Call evaluate exactly once per symmetry to lock in the sorted layout.
            evaluate(); 

            // --- 8. FREEZE ---

            State current_state = {sorted_verts, sorted_edges, final_faces, best_final_instances};
    
            // Compare states in C++ first
            if (!best_state_ptr || current_state < *best_state_ptr) {
                best_state_ptr = std::make_unique<State>(std::move(current_state));
            }
        }
    }

    // 3. Cast the object back to a tuple before returning
    // return best_key;
    return internal_freeze(best_state_ptr->v, best_state_ptr->e, best_state_ptr->f, best_state_ptr->i);
}

// --- Slicing and Instance Rebuilding Core ---
py::tuple split_and_rebuild_cpp(
    const std::vector<std::pair<int, int>>& edges,
    const std::vector<std::vector<int>>& faces,
    const std::vector<std::vector<std::vector<std::pair<int, int>>>>& instances,
    const std::vector<int>& vertex_sides,
    const std::map<int, int>& intersections
) {
    std::vector<std::pair<int, int>> new_edges;
    std::map<std::pair<int, int>, int> edge_lookup;
    
    // Helper to fetch or create unique undirected edges
    auto get_e = [&](int u, int v) {
        std::pair<int, int> key = {std::min(u, v), std::max(u, v)};
        auto it = edge_lookup.find(key);
        if (it != edge_lookup.end()) return it->second;
        int idx = new_edges.size();
        new_edges.push_back(key);
        edge_lookup[key] = idx;
        return idx;
    };

    // --- Step 1: Split Topology (Edges) ---
    std::map<int, std::vector<int>> old_edge_map;
    for (size_t e_idx = 0; e_idx < edges.size(); ++e_idx) {
        auto it = intersections.find(e_idx);
        if (it != intersections.end()) {
            int iv = it->second;
            old_edge_map[e_idx] = {get_e(edges[e_idx].first, iv), get_e(iv, edges[e_idx].second)};
        } else {
            old_edge_map[e_idx] = {get_e(edges[e_idx].first, edges[e_idx].second)};
        }
    }

    // --- Step 2: Split Topology (Faces) ---
    std::vector<std::vector<int>> new_faces;
    std::map<int, std::vector<int>> old_face_map;

    for (size_t f_idx = 0; f_idx < faces.size(); ++f_idx) {
        const auto& face_edges = faces[f_idx];
        if (face_edges.empty()) continue;

        // Reconstruct Vertex Loop
        std::vector<int> loop_verts;
        int e_last = face_edges.back();
        int e_first = face_edges.front();
        int v_start = -1;
        
        int e_first_v0 = edges[e_first].first;
        int e_first_v1 = edges[e_first].second;
        if (edges[e_last].first == e_first_v0 || edges[e_last].first == e_first_v1) {
            v_start = edges[e_last].first;
        } else {
            v_start = edges[e_last].second;
        }

        int curr_v = v_start;
        for (int e_idx : face_edges) {
            const auto& segments = old_edge_map[e_idx];
            int v_def_1 = edges[e_idx].first;

            if (curr_v == v_def_1) {
                for (int seg_idx : segments) {
                    int u = new_edges[seg_idx].first;
                    int v = new_edges[seg_idx].second;
                    int next_v = (u == curr_v) ? v : u;
                    loop_verts.push_back(curr_v);
                    curr_v = next_v;
                }
            } else {
                for (auto it = segments.rbegin(); it != segments.rend(); ++it) {
                    int seg_idx = *it;
                    int u = new_edges[seg_idx].first;
                    int v = new_edges[seg_idx].second;
                    int next_v = (u == curr_v) ? v : u;
                    loop_verts.push_back(curr_v);
                    curr_v = next_v;
                }
            }
        }

        // Check if bifurcation is needed
        bool has_pos = false, has_neg = false;
        std::vector<int> loop_sides(loop_verts.size());
        for (size_t i = 0; i < loop_verts.size(); ++i) {
            int s = vertex_sides[loop_verts[i]];
            loop_sides[i] = s;
            if (s == 1) has_pos = true;
            if (s == -1) has_neg = true;
        }

        if (!has_pos || !has_neg) {
            std::vector<int> f_new;
            for (size_t i = 0; i < loop_verts.size(); ++i) {
                f_new.push_back(get_e(loop_verts[i], loop_verts[(i + 1) % loop_verts.size()]));
            }
            int new_f_idx = new_faces.size();
            new_faces.push_back(f_new);
            old_face_map[f_idx].push_back(new_f_idx);
            continue;
        }

        // Bifurcate Face
        int entry_idx = -1, exit_idx = -1;
        int n = loop_verts.size();
        for (int i = 0; i < n; ++i) {
            int curr_s = loop_sides[i];
            int nxt_s = loop_sides[(i + 1) % n];
            if (curr_s <= 0 && nxt_s == 1) entry_idx = i;
            else if (curr_s >= 0 && nxt_s == -1) exit_idx = i;
        }

        std::vector<int> left_chain, right_chain;
        int curr = exit_idx;
        while (true) {
            left_chain.push_back(loop_verts[curr]);
            if (curr == entry_idx) break;
            curr = (curr + 1) % n;
        }
        curr = entry_idx;
        while (true) {
            right_chain.push_back(loop_verts[curr]);
            if (curr == exit_idx) break;
            curr = (curr + 1) % n;
        }

        std::vector<int> f_left, f_right;
        for (size_t i = 0; i < left_chain.size() - 1; ++i) f_left.push_back(get_e(left_chain[i], left_chain[i + 1]));
        int crease = get_e(left_chain.back(), left_chain.front());
        f_left.push_back(crease);

        for (size_t i = 0; i < right_chain.size() - 1; ++i) f_right.push_back(get_e(right_chain[i], right_chain[i + 1]));
        f_right.push_back(crease);

        int idx_l = new_faces.size();
        new_faces.push_back(f_left);
        int idx_r = new_faces.size();
        new_faces.push_back(f_right);
        old_face_map[f_idx] = {idx_l, idx_r};
    }

    // --- Step 3: Rebuild Instances (Pass 1 - Mapping) ---
    std::vector<std::vector<std::vector<std::pair<int, int>>>> new_instances(new_faces.size());
    std::map<std::pair<int, int>, std::vector<std::pair<int, int>>> i_map;

    for (size_t old_f_idx = 0; old_f_idx < instances.size(); ++old_f_idx) {
        const auto& old_face_insts = instances[old_f_idx];
        const auto& new_f_indices = old_face_map[old_f_idx];

        for (size_t old_i_idx = 0; old_i_idx < old_face_insts.size(); ++old_i_idx) {
            const auto& old_inst = old_face_insts[old_i_idx];

            for (int nf_idx : new_f_indices) {
                const auto& current_new_face = new_faces[nf_idx];
                std::vector<std::pair<int, int>> new_inst(current_new_face.size(), {-1, -1});

                // Inherited connections
                for (size_t old_slot = 0; old_slot < old_inst.size(); ++old_slot) {
                    int old_e = faces[old_f_idx][old_slot];
                    std::pair<int, int> connection = old_inst[old_slot];

                    const auto& segments = old_edge_map[old_e];
                    for (int seg : segments) {
                        auto it = std::find(current_new_face.begin(), current_new_face.end(), seg);
                        if (it != current_new_face.end()) {
                            int slot = std::distance(current_new_face.begin(), it);
                            new_inst[slot] = connection;
                        }
                    }
                }

                // Internal Crease Mapping
                if (new_f_indices.size() > 1) {
                    int other_nf = (nf_idx == new_f_indices[0]) ? new_f_indices[1] : new_f_indices[0];
                    int crease_e = -1;
                    for (int e1 : current_new_face) {
                        auto it = std::find(new_faces[other_nf].begin(), new_faces[other_nf].end(), e1);
                        if (it != new_faces[other_nf].end()) {
                            crease_e = e1;
                            break;
                        }
                    }
                    if (crease_e != -1) {
                        auto it = std::find(current_new_face.begin(), current_new_face.end(), crease_e);
                        int crease_slot = std::distance(current_new_face.begin(), it);
                        new_inst[crease_slot] = {-2, other_nf}; // -2 denotes INTERNAL
                    }
                }

                int new_ni_idx = new_instances[nf_idx].size();
                new_instances[nf_idx].push_back(new_inst);
                i_map[{old_f_idx, old_i_idx}].push_back({nf_idx, new_ni_idx});
            }
        }
    }

    // --- Step 4: Rebuild Instances (Pass 2 - Pointer Resolution) ---
    for (size_t nf_idx = 0; nf_idx < new_instances.size(); ++nf_idx) {
        for (size_t ni_idx = 0; ni_idx < new_instances[nf_idx].size(); ++ni_idx) {
            auto& inst = new_instances[nf_idx][ni_idx];
            for (size_t slot = 0; slot < inst.size(); ++slot) {
                auto& conn = inst[slot];
                if (conn.first == -1) continue; // Leave boundaries alone

                if (conn.first == -2) {
                    // Resolve internal connection
                    int other_nf = conn.second;
                    conn = {other_nf, (int)ni_idx};
                } else {
                    auto it = i_map.find(conn);
                    if (it == i_map.end() || it->second.empty()) {
                        conn = {-1, -1};
                        continue;
                    }
                    const auto& targets = it->second;
                    if (targets.size() == 1) {
                        conn = targets[0];
                    } else {
                        // Split Target Resolution
                        int current_edge = new_faces[nf_idx][slot];
                        bool match_found = false;
                        for (const auto& target : targets) {
                            const auto& target_face = new_faces[target.first];
                            if (std::find(target_face.begin(), target_face.end(), current_edge) != target_face.end()) {
                                conn = target;
                                match_found = true;
                                break;
                            }
                        }
                        if (!match_found) conn = {-1, -1};
                    }
                }
            }
        }
    }

    return py::make_tuple(new_edges, new_faces, new_instances);
}


// --- Flatten Core ---
py::tuple flatten_cpp(const std::vector<Vertex4D>& vertices,
                      const std::vector<std::pair<int, int>>& edges,
                      const std::vector<std::vector<int>>& faces,
                      const std::vector<std::vector<std::vector<std::pair<int, int>>>>& instances) {
    
    // 1. Deduplicate Vertices (O(N log N) using std::map)
    std::vector<Vertex4D> unique_verts;
    std::map<Vertex4D, int> v_dict;
    std::vector<int> v_map(vertices.size());
    
    for (size_t i = 0; i < vertices.size(); ++i) {
        auto it = v_dict.find(vertices[i]);
        if (it == v_dict.end()) {
            int new_idx = unique_verts.size();
            v_dict[vertices[i]] = new_idx;
            unique_verts.push_back(vertices[i]);
            v_map[i] = new_idx;
        } else {
            v_map[i] = it->second;
        }
    }
    
    // 2. Deduplicate Edges
    std::vector<std::pair<int, int>> unique_edges;
    std::map<std::pair<int, int>, int> e_dict;
    std::vector<int> e_map(edges.size(), -1);
    
    for (size_t i = 0; i < edges.size(); ++i) {
        int new_v1 = v_map[edges[i].first];
        int new_v2 = v_map[edges[i].second];
        // Canonicalize the edge by ensuring smaller vertex index is first
        std::pair<int, int> canon_e = {std::min(new_v1, new_v2), std::max(new_v1, new_v2)};
        
        auto it = e_dict.find(canon_e);
        if (it == e_dict.end()) {
            int new_idx = unique_edges.size();
            e_dict[canon_e] = new_idx;
            unique_edges.push_back(canon_e);
            e_map[i] = new_idx;
        } else {
            e_map[i] = it->second;
        }
    }
    
    // 3. Deduplicate Faces
    std::vector<std::vector<int>> unique_faces;
    std::map<std::vector<int>, int> f_dict;
    std::vector<int> f_map(faces.size(), -1);
    
    struct FaceTransform { bool was_rev; int shift; };
    std::vector<FaceTransform> f_transforms(faces.size());
    
    for (size_t i = 0; i < faces.size(); ++i) {
        std::vector<int> temp_face;
        for (int e_idx : faces[i]) {
            if (e_idx >= 0 && e_idx < (int)e_map.size() && e_map[e_idx] != -1) {
                temp_face.push_back(e_map[e_idx]);
            }
        }
        
        // Reuse the exact same rotational canonicalization from canonicalize_cpp!
        auto result = canonicalize_face_indices(temp_face);
        std::vector<int> canon_f = std::get<0>(result);
        bool was_rev = std::get<1>(result);
        int shift = std::get<2>(result);
        
        auto it = f_dict.find(canon_f);
        if (it == f_dict.end()) {
            int new_idx = unique_faces.size();
            f_dict[canon_f] = new_idx;
            unique_faces.push_back(canon_f);
            f_map[i] = new_idx;
        } else {
            f_map[i] = it->second;
        }
        f_transforms[i] = {was_rev, shift};
    }
    
    // 4. Update Instances
    std::map<std::pair<int, int>, std::pair<int, int>> i_map;
    std::vector<std::vector<std::vector<std::pair<int, int>>>> new_instances(unique_faces.size());
    
    for (size_t old_f_idx = 0; old_f_idx < instances.size(); ++old_f_idx) {
        int new_f_idx = f_map[old_f_idx];
        bool was_rev = f_transforms[old_f_idx].was_rev;
        int shift = f_transforms[old_f_idx].shift;
        
        for (size_t old_i_idx = 0; old_i_idx < instances[old_f_idx].size(); ++old_i_idx) {
            int new_i_idx = new_instances[new_f_idx].size();
            i_map[{old_f_idx, old_i_idx}] = {new_f_idx, new_i_idx};
            
            std::vector<std::pair<int, int>> curr_inst = instances[old_f_idx][old_i_idx];
            
            // Align the instance connections using the same shifts the face underwent
            if (was_rev) std::reverse(curr_inst.begin(), curr_inst.end());
            if (!curr_inst.empty()) {
                std::rotate(curr_inst.begin(), curr_inst.begin() + shift, curr_inst.end());
            }
            
            new_instances[new_f_idx].push_back(curr_inst);
        }
    }
    
    // Remap pointers
    for (size_t f_idx = 0; f_idx < new_instances.size(); ++f_idx) {
        for (size_t i_idx = 0; i_idx < new_instances[f_idx].size(); ++i_idx) {
            for (size_t slot = 0; slot < new_instances[f_idx][i_idx].size(); ++slot) {
                std::pair<int, int> conn = new_instances[f_idx][i_idx][slot];
                if (conn.first != -1) {
                    new_instances[f_idx][i_idx][slot] = i_map[conn];
                }
            }
        }
    }
    
    return py::make_tuple(unique_verts, unique_edges, unique_faces, new_instances);
}

// ---  The PyBind11 Bridge ---
PYBIND11_MODULE(math225_core, m) {
    m.doc() = "C++ Math Foundation for SEARCH22.5";
    py::class_<Fraction>(m, "Fraction")
        .def(py::init([](int64_t n, int64_t d) { return Fraction(n, d); }), py::arg("num"), py::arg("den") = 1)
        // Copy Constructor
        .def(py::init([](const Fraction &other) { return Fraction(other.num, other.den); }))
        .def_readwrite("num", &Fraction::num)
        .def_readwrite("den", &Fraction::den)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(py::self == py::self)
        .def(py::self < py::self)
        .def(py::self > py::self)
        .def(-py::self)
        // Support for `int + Fraction` and `int * Fraction` in Python
        .def("__radd__", [](const Fraction &self, int64_t other) { return self + Fraction(other, 1); })
        .def("__rsub__", [](const Fraction &self, int64_t other) { return Fraction(other, 1) - self; })
        .def("__rmul__", [](const Fraction &self, int64_t other) { return self * Fraction(other, 1); })
        .def("__float__", &Fraction::to_float)
        .def("__hash__", [](const Fraction& f) {
            // Hash relies on simplified num/den directly
            return py::hash(py::make_tuple(f.num, f.den)); 
        })
        .def("__repr__", [](const Fraction& f) {
            return "Fraction(" + std::to_string(f.num) + ", " + std::to_string(f.den) + ")";
        })
        .def("__str__", [](const Fraction& f) {
            return std::to_string(f.num) + "/" + std::to_string(f.den);
        });

    py::implicitly_convertible<py::int_, Fraction>();

    py::class_<AplusBsqrt2>(m, "AplusBsqrt2")
        .def(py::init<Fraction, Fraction>(), py::arg("A"), py::arg("B") = Fraction(0, 1))
        .def_readwrite("A", &AplusBsqrt2::A)
        .def_readwrite("B", &AplusBsqrt2::B)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(py::self == py::self)
        .def(py::self < py::self)
        .def(py::self > py::self)
        .def(-py::self)
        .def("sign", &AplusBsqrt2::sign)
        .def("__float__", &AplusBsqrt2::to_float)
        .def("__hash__", [](const AplusBsqrt2& v) {
            return py::hash(py::make_tuple(v.A.num, v.A.den, v.B.num, v.B.den));
        })
        .def("__repr__", [](const AplusBsqrt2& v) {
            return "AplusBsqrt2(Fraction(" + std::to_string(v.A.num) + ", " + std::to_string(v.A.den) + "), Fraction(" + std::to_string(v.B.num) + ", " + std::to_string(v.B.den) + "))";
        })
        .def("__mul__", [](const AplusBsqrt2& self, py::object other) -> py::object {
            if (py::isinstance<AplusBsqrt2>(other)) {
                return py::cast(self * other.cast<AplusBsqrt2>());
            } else if (py::isinstance<Fraction>(other)) {
                return py::cast(self * AplusBsqrt2(other.cast<Fraction>()));
            } else if (py::isinstance<py::int_>(other)) {
                return py::cast(self * AplusBsqrt2(Fraction(other.cast<int64_t>(), 1)));
            }
            return py::reinterpret_borrow<py::object>(Py_NotImplemented); 
        });

    py::implicitly_convertible<Fraction, AplusBsqrt2>();

    py::class_<Vertex4D>(m, "Vertex4D")
        .def(py::init<Fraction, Fraction, Fraction, Fraction>())
        .def_readwrite("x", &Vertex4D::x).def_readwrite("y", &Vertex4D::y)
        .def_readwrite("z", &Vertex4D::z).def_readwrite("w", &Vertex4D::w)
        .def(py::self == py::self)
        .def("__hash__", [](const Vertex4D& v) {
            return py::hash(py::make_tuple(v.x, v.y, v.z, v.w));
        })
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def("__mul__", [](const Vertex4D& self, const Fraction& f) { return self * f; })
        .def("__mul__", [](const Vertex4D& self, const AplusBsqrt2& a) { return self * a; })
        .def("to_cartesian", &Vertex4D::to_cartesian)
        .def("dot_product", &Vertex4D::dot_product)
        .def("angle_to", &Vertex4D::angle_to)
        .def("__str__", [](const Vertex4D &self) {
            return "(" + py::str(py::cast(self.x)).cast<std::string>() + ", " + 
                        py::str(py::cast(self.y)).cast<std::string>() + ", " + 
                        py::str(py::cast(self.z)).cast<std::string>() + ", " + 
                        py::str(py::cast(self.w)).cast<std::string>() + ")";
        })
        .def("__repr__", [](const Vertex4D &self) {
            return "Vertex4D(" + py::str(py::cast(self.x)).cast<std::string>() + ", " + 
                                py::str(py::cast(self.y)).cast<std::string>() + ", " + 
                                py::str(py::cast(self.z)).cast<std::string>() + ", " + 
                                py::str(py::cast(self.w)).cast<std::string>() + ")";
        });

    m.def("reflect", &reflect);
    m.def("reflect_group", [](const Vertex4D& v1, const Vertex4D& v2, const std::vector<Vertex4D>& group) {
        std::vector<Vertex4D> result;
        for (const auto& p : group) result.push_back(reflect(v1, v2, p));
        return result;
    });

    m.def("canonicalize_cpp", &canonicalize_cpp);
    m.def("flatten_cpp", &flatten_cpp);
    m.def("split_and_rebuild_cpp", &split_and_rebuild_cpp);
}