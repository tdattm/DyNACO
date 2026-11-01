#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
namespace py=pybind11;
struct Replay {
 int32_t n,k=32; bool extend_ls=true;
 std::vector<float> coords; std::vector<int32_t> nn_list;
 std::vector<bool> processed; py::list events;
 float dist(int32_t a,int32_t b) const {
  const size_t ia=static_cast<size_t>(a)*2,ib=static_cast<size_t>(b)*2;
  const float dx=coords[ib+0]-coords[ia+0],dy=coords[ib+1]-coords[ia+1];
  return std::sqrt(dx*dx+dy*dy);
 }
int32_t get_succ(int32_t node, const std::vector<int32_t> &route,
                            const std::vector<int32_t> &positions) const {
  int32_t pos = positions[node];
  return route[(pos + 1) % n];
}
int32_t get_pred(int32_t node, const std::vector<int32_t> &route,
                            const std::vector<int32_t> &positions) const {
  int32_t pos = positions[node];
  return route[(pos - 1 + n) % n];
}
void flip_route_section(int32_t start_node, int32_t end_node,
                                   std::vector<int32_t> &route,
                                   std::vector<int32_t> &positions) {
  int32_t first = positions[start_node];
  int32_t last = positions[end_node];

  if (first > last) {
    std::swap(first, last);
  }

  int32_t segment_length = last - first;
  int32_t remaining_length = n - segment_length;

  if (segment_length <= remaining_length) {
    // Flip the segment
    int32_t left = first;
    int32_t right = last - 1;
    while (left < right) {
      std::swap(route[left], route[right]);
      ++left;
      --right;
    }
    for (int32_t i = first; i < last; ++i) {
      positions[route[i]] = i;
    }
  } else {
    // Flip the other segment (wrap around)
    int32_t first_adj = (first > 0) ? first - 1 : n - 1;
    int32_t last_adj = last % n;
    std::swap(first_adj, last_adj);

    int32_t l = first_adj;
    int32_t r = last_adj;
    int32_t i = 0;
    int32_t j = n - first_adj + last_adj + 1;

    while (i < j) {
      std::swap(route[l], route[r]);
      positions[route[l]] = l;
      positions[route[r]] = r;
      l = (l + 1) % n;
      r = (r - 1 + n) % n;
      ++i;
      --j;
    }
  }
}
float two_opt_nn(std::vector<int32_t> &route,
                            std::vector<int32_t> &positions,
                            std::vector<int32_t> &checklist) {
  int32_t changes_count = 0;
  float total_change = 0.0f;
  size_t checklist_pos = 0;

  while (checklist_pos < checklist.size()) {
    int32_t a = checklist[checklist_pos++];
    if (a < 0 || a >= n)
      continue;

    int32_t a_next = get_succ(a, route, positions);
    int32_t a_prev = get_pred(a, route, positions);

    float dist_a_to_next = dist(a, a_next);
    float dist_a_to_prev = dist(a_prev, a);

    float max_diff = 0.0f;
    int32_t best_move[4] = {-1, -1, -1, -1};

    // Check moves with a -> a_next edge
    for (int32_t j = 0; j < k; ++j) {
      int32_t b = nn_list[a * k + j];
      if (b < 0 || b >= n)
        break;

      float dist_ab = dist(a, b);
      if (dist_a_to_next > dist_ab) {
        int32_t b_next = get_succ(b, route, positions);
        float diff =
            dist_a_to_next + dist(b, b_next) - dist_ab - dist(a_next, b_next);
        if (diff > max_diff) {
          best_move[0] = a_next;
          best_move[1] = b_next;
          best_move[2] = a;
          best_move[3] = b;
          max_diff = diff;
        }
      } else {
        break;
      }
    }

    // Check moves with a_prev -> a edge
    for (int32_t j = 0; j < k; ++j) {
      int32_t b = nn_list[a * k + j];
      if (b < 0 || b >= n)
        break;

      float dist_ab = dist(a, b);
      if (dist_a_to_prev > dist_ab) {
        int32_t b_prev = get_pred(b, route, positions);
        float diff =
            dist_a_to_prev + dist(b_prev, b) - dist_ab - dist(a_prev, b_prev);
        if (diff > max_diff) {
          best_move[0] = a;
          best_move[1] = b;
          best_move[2] = a_prev;
          best_move[3] = b_prev;
          max_diff = diff;
        }
      } else {
        break;
      }
    }

    if (max_diff > 0) {
      flip_route_section(best_move[0], best_move[1], route, positions);
      ++changes_count;
      total_change -= max_diff;

      // if extend_ls, then add endpoints to checklist
      if (extend_ls) {
        for (int32_t i = 0; i < 4; ++i) {
          int32_t node = best_move[i];
          if (std::find(checklist.begin(), checklist.end(), node) ==
              checklist.end()) {
            checklist.push_back(node);
          }
        }
      }
    }
  }

  return total_change;
}
float two_opt_logged(std::vector<int32_t> &route,
                            std::vector<int32_t> &positions,
                            std::vector<int32_t> &checklist) {
  int32_t changes_count = 0;
  float total_change = 0.0f;
  size_t checklist_pos = 0;

  while (checklist_pos < checklist.size()) {
    int32_t a = checklist[checklist_pos++];
    py::dict event; py::list evals, prunes, insertions;
    event["cursor"] = checklist_pos - 1; event["node"] = a;
    event["checklist_size_before"] = checklist.size();
    processed[a] = true;
    if (a < 0 || a >= n)
      continue;

    int32_t a_next = get_succ(a, route, positions);
    int32_t a_prev = get_pred(a, route, positions);

    float dist_a_to_next = dist(a, a_next);
    float dist_a_to_prev = dist(a_prev, a);

    float max_diff = 0.0f;
    int32_t best_move[4] = {-1, -1, -1, -1};

    int direction = 0;
    // Check moves with a -> a_next edge
    for (int32_t j = 0; j < k; ++j) {
      int32_t b = nn_list[a * k + j];
      if (b < 0 || b >= n) {
        py::dict p; p["direction"] = direction; p["rank"] = j;
        p["reason"] = "invalid_neighbor"; prunes.append(p); break;
      }

      float dist_ab = dist(a, b);
      if (dist_a_to_next > dist_ab) {
        int32_t b_next = get_succ(b, route, positions);
        float diff =
            dist_a_to_next + dist(b, b_next) - dist_ab - dist(a_next, b_next);
        py::dict e; e["direction"] = direction; e["rank"] = j;
        e["b"] = b; e["gain_f32"] = diff; e["selected_at_this_point"] = diff > max_diff;
        evals.append(e);
        if (diff > max_diff) {
          best_move[0] = a_next;
          best_move[1] = b_next;
          best_move[2] = a;
          best_move[3] = b;
          max_diff = diff;
        }
      } else {
        py::dict p; p["direction"] = direction; p["rank"] = j; p["b"] = b;
        p["reason"] = "distance_early_break"; p["remaining_including_boundary"] = k-j;
        p["edge_distance_f32"] = direction == 0 ? dist_a_to_next : dist_a_to_prev;
        p["neighbor_distance_f32"] = dist_ab; prunes.append(p); break;
      }
    }

    direction = 1;
    // Check moves with a_prev -> a edge
    for (int32_t j = 0; j < k; ++j) {
      int32_t b = nn_list[a * k + j];
      if (b < 0 || b >= n) {
        py::dict p; p["direction"] = direction; p["rank"] = j;
        p["reason"] = "invalid_neighbor"; prunes.append(p); break;
      }

      float dist_ab = dist(a, b);
      if (dist_a_to_prev > dist_ab) {
        int32_t b_prev = get_pred(b, route, positions);
        float diff =
            dist_a_to_prev + dist(b_prev, b) - dist_ab - dist(a_prev, b_prev);
        py::dict e; e["direction"] = direction; e["rank"] = j;
        e["b"] = b; e["gain_f32"] = diff; e["selected_at_this_point"] = diff > max_diff;
        evals.append(e);
        if (diff > max_diff) {
          best_move[0] = a;
          best_move[1] = b;
          best_move[2] = a_prev;
          best_move[3] = b_prev;
          max_diff = diff;
        }
      } else {
        py::dict p; p["direction"] = direction; p["rank"] = j; p["b"] = b;
        p["reason"] = "distance_early_break"; p["remaining_including_boundary"] = k-j;
        p["edge_distance_f32"] = direction == 0 ? dist_a_to_next : dist_a_to_prev;
        p["neighbor_distance_f32"] = dist_ab; prunes.append(p); break;
      }
    }

    event["evaluations"] = evals; event["prunes"] = prunes;
    event["selected_gain_f32"] = max_diff;
    event["selected_move"] = std::vector<int32_t>(best_move, best_move+4);
    if (max_diff > 0) {
      auto before_route = route;
      flip_route_section(best_move[0], best_move[1], route, positions);
      ++changes_count;
      py::list updates;
      for(int z=0;z<n;++z) if(route[z]!=before_route[z]) updates.append(py::make_tuple(z,route[z]));
      event["route_updates"] = updates;
      for(int z=0;z<n;++z) if(positions[route[z]]!=z) throw std::runtime_error("position invariant");
      total_change -= max_diff;

      // if extend_ls, then add endpoints to checklist
      if (extend_ls) {
        for (int32_t i = 0; i < 4; ++i) {
          int32_t node = best_move[i];
          py::dict ins; ins["node"] = node; ins["already_processed"] = bool(processed[node]);
          ins["inserted"] = std::find(checklist.begin(),checklist.end(),node)==checklist.end();
          insertions.append(ins);
          if (std::find(checklist.begin(), checklist.end(), node) ==
              checklist.end()) {
            checklist.push_back(node);
          }
        }
      }
    }
    event["insertions"] = insertions;
    event["checklist_size_after"] = checklist.size();
    events.append(event);
    if(events.size()>size_t(n)) throw std::runtime_error("bounded log exceeded");
  }

  return total_change;
}
};
py::dict run(py::array_t<float,py::array::c_style> x,
             std::vector<int32_t> route,std::vector<int32_t> checklist,
             py::array_t<int32_t,py::array::c_style> nn,bool logging) {
 Replay s; s.n=route.size();
 if(x.ndim()!=2 || x.shape(0)!=s.n || x.shape(1)!=2 || nn.ndim()!=2 || nn.shape(0)!=s.n || nn.shape(1)!=32)
  throw std::runtime_error("shape mismatch");
 s.coords.assign(x.data(),x.data()+s.n*2); s.nn_list.assign(nn.data(),nn.data()+s.n*32);
 std::vector<int32_t> positions(s.n,-1); s.processed.assign(s.n,false);
 for(int i=0;i<s.n;++i) {if(route[i]<0||route[i]>=s.n||positions[route[i]]!=-1) throw std::runtime_error("permutation");positions[route[i]]=i;}
 std::vector<bool> seen(s.n,false);
 for(int u:checklist) {if(u<0||u>=s.n||seen[u]) throw std::runtime_error("checklist");seen[u]=true;}
 for(int u:s.nn_list) if(u<0||u>=s.n) throw std::runtime_error("candidate");
 float change=logging?s.two_opt_logged(route,positions,checklist):s.two_opt_nn(route,positions,checklist);
 py::dict out;out["route"]=route;out["checklist"]=checklist;out["total_change_f32"]=change;out["events"]=s.events;return out;
}
PYBIND11_MODULE(m0_native,m) {m.def("run",&run);}
