"""Build a test-only replay from literal baseline method bodies; never run ACO."""
import json
import os
from pathlib import Path
from setuptools import Extension, setup
import pybind11

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/r3/m0_srr_replay'
OUT.mkdir(exist_ok=False, parents=True)
src = (ROOT / 'src/mfaco_train.cpp').read_text()

def method(name):
    start = src.index(name)
    start = src.rfind('\n', 0, start) + 1
    brace = src.index('{', start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (src[end] == '{') - (src[end] == '}')
        end += 1
    return src[start:end].replace('MFACO_TSP::', '')

methods = [method(n) for n in ('MFACO_TSP::get_succ(', 'MFACO_TSP::get_pred(',
                              'MFACO_TSP::flip_route_section(', 'MFACO_TSP::two_opt_nn(')]
logged = methods[-1].replace('two_opt_nn(', 'two_opt_logged(', 1)
replacements = {
    'int32_t a = checklist[checklist_pos++];': '''int32_t a = checklist[checklist_pos++];
    py::dict event; py::list evals, prunes, insertions;
    event["cursor"] = checklist_pos - 1; event["node"] = a;
    event["checklist_size_before"] = checklist.size();
    processed[a] = true;''',
    'if (b < 0 || b >= n)\n        break;': '''if (b < 0 || b >= n) {
        py::dict p; p["direction"] = direction; p["rank"] = j;
        p["reason"] = "invalid_neighbor"; prunes.append(p); break;
      }''',
    '// Check moves with a -> a_next edge': 'int direction = 0;\n    // Check moves with a -> a_next edge',
    '// Check moves with a_prev -> a edge': 'direction = 1;\n    // Check moves with a_prev -> a edge',
    'if (diff > max_diff) {': '''py::dict e; e["direction"] = direction; e["rank"] = j;
        e["b"] = b; e["gain_f32"] = diff; e["selected_at_this_point"] = diff > max_diff;
        evals.append(e);
        if (diff > max_diff) {''',
    '} else {\n        break;\n      }': '''} else {
        py::dict p; p["direction"] = direction; p["rank"] = j; p["b"] = b;
        p["reason"] = "distance_early_break"; p["remaining_including_boundary"] = k-j;
        p["edge_distance_f32"] = direction == 0 ? dist_a_to_next : dist_a_to_prev;
        p["neighbor_distance_f32"] = dist_ab; prunes.append(p); break;
      }''',
    'if (max_diff > 0) {': '''event["evaluations"] = evals; event["prunes"] = prunes;
    event["selected_gain_f32"] = max_diff;
    event["selected_move"] = std::vector<int32_t>(best_move, best_move+4);
    if (max_diff > 0) {
      auto before_route = route;''',
    '++changes_count;': '''++changes_count;
      py::list updates;
      for(int z=0;z<n;++z) if(route[z]!=before_route[z]) updates.append(py::make_tuple(z,route[z]));
      event["route_updates"] = updates;
      for(int z=0;z<n;++z) if(positions[route[z]]!=z) throw std::runtime_error("position invariant");''',
    'int32_t node = best_move[i];': '''int32_t node = best_move[i];
          py::dict ins; ins["node"] = node; ins["already_processed"] = bool(processed[node]);
          ins["inserted"] = std::find(checklist.begin(),checklist.end(),node)==checklist.end();
          insertions.append(ins);''',
    '\n  }\n\n  return total_change;': '''
    event["insertions"] = insertions;
    event["checklist_size_after"] = checklist.size();
    events.append(event);
    if(events.size()>size_t(n)) throw std::runtime_error("bounded log exceeded");
  }

  return total_change;''',
}
for old, new in replacements.items():
    if old not in logged:
        raise AssertionError(old)
    logged = logged.replace(old, new)
header = '''#include <pybind11/pybind11.h>
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
'''
tail = '''
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
'''
(OUT / 'replayer.cpp').write_text(header + '\n'.join(methods) + '\n' + logged + tail)
(OUT / 'instrumentation_replacements.json').write_text(json.dumps(replacements, indent=2))
setup(name='m0_native', version='1.0', ext_modules=[Extension('m0_native',
    [str(OUT/'replayer.cpp')], include_dirs=[pybind11.get_include()], language='c++',
    extra_compile_args=['-O3','-std=c++17'])],
    script_args=['build_ext','--build-lib',str(ROOT/'.r3-build/m0'),
                 '--build-temp',str(ROOT/'.r3-build/m0-objects')])
