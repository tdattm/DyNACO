// Independent streaming diagnostic. Does not include or call the production solver.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>
namespace py = pybind11;
using F = py::array_t<float, py::array::c_style | py::array::forcecast>;
using I = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
uint64_t key(int u,int v) { return (uint64_t(uint32_t(u))<<32)|uint32_t(v); }
struct Record {
  int i,j; double gain; float native_gain; std::array<int,4> ranks;
  std::vector<bool> membership, scan;
  bool checklist_member,checklist_scan;
};
py::dict scan(F coords, I tour, I neighbors, I widths, I checklist, int cap=32) {
  auto start=Clock::now();
  auto x=coords.unchecked<2>(); auto t=tour.unchecked<1>();
  auto nn=neighbors.unchecked<2>(); auto w=widths.unchecked<2>(); auto cl=checklist.unchecked<1>();
  const int n=x.shape(0), k=nn.shape(1), p=w.shape(0);
  if(x.shape(1)!=2 || t.shape(0)!=n || nn.shape(0)!=n || w.shape(1)!=n || n<4 || k<1 || k>64 || p<1 || cap<0 || cap>10000)
    throw std::invalid_argument("invalid shapes, n, k, policy count, or cap");
  if(n>200 && cap>64) throw std::invalid_argument("bounded detailed storage required above n=200");
  auto d=[&](int u,int v) { double dx=double(x(u,0))-double(x(v,0)),dy=double(x(u,1))-double(x(v,1));return std::sqrt(dx*dx+dy*dy); };
  auto df=[&](int u,int v) { float dx=x(v,0)-x(u,0),dy=x(v,1)-x(u,1);return std::sqrt(dx*dx+dy*dy); };
  std::vector<int> seen(n),pos(n),prefix_next(n),prefix_prev(n);
  std::vector<bool> in_cl(n,false);
  for(int u=0;u<n;++u) if(!std::isfinite(x(u,0)) || !std::isfinite(x(u,1))) throw std::invalid_argument("nonfinite coordinates");
  for(int i=0;i<n;++i) {int u=t(i);if(u<0||u>=n||seen[u]++)throw std::invalid_argument("invalid tour permutation");pos[u]=i;}
  for(int i=0;i<cl.shape(0);++i) {if(cl(i)<0||cl(i)>=n)throw std::invalid_argument("invalid checklist");in_cl[cl(i)]=true;}
  std::unordered_map<uint64_t,int> rank; rank.reserve(n*k*2);
  for(int u=0;u<n;++u) {
    double last=-1;
    for(int j=0;j<k;++j) {int v=nn(u,j);if(v<0||v>=n||v==u||!rank.emplace(key(u,v),j+1).second)throw std::invalid_argument("invalid, duplicate or self neighbor");
      double dd=d(u,v);if(dd+1e-14<last)throw std::invalid_argument("neighbors not sorted by exact Euclidean distance");last=dd;}
    for(int z=0;z<p;++z) if(w(z,u)<0||w(z,u)>k)throw std::invalid_argument("policy width outside candidate universe");
    int succ=t((pos[u]+1)%n),pred=t((pos[u]+n-1)%n);
    // Literal early-break semantics, including float32 equality and any rounding inversion.
    while(prefix_next[u]<k && df(u,succ)>df(u,nn(u,prefix_next[u])))++prefix_next[u];
    while(prefix_prev[u]<k && df(pred,u)>df(u,nn(u,prefix_prev[u])))++prefix_prev[u];
  }
  double length=0;for(int i=0;i<n;++i)length+=d(t(i),t((i+1)%n));
  const double threshold=std::max(1e-8,1e-6*length);
  double setup=elapsed(start);auto loop=Clock::now();
  int64_t valid=0,improving=0,positive_marginal=0,zero=0,negative=0;
  std::vector<int64_t> member_count(p),scan_count(p),miss_hist(7),rank_hist(k+2);
  std::vector<double> member_gain(p),scan_gain(p),max_scan_gain(p),node_gain(n);
  double gain_total=0,max_missed=0,classification_seconds=0;int64_t cl_member_count=0,cl_scan_count=0;
  std::vector<std::array<double,3>> signed_sample;
  double cl_member_gain=0,cl_scan_gain=0;
  std::vector<Record> top,sample; top.reserve(cap);sample.reserve(cap);
  auto rnk=[&](int u,int v) {auto it=rank.find(key(u,v));return it==rank.end()?k+1:it->second;};
  for(int i=0;i<n;++i)for(int j=i+2;j<n;++j) {
    if(i==0&&j==n-1)continue;
    ++valid;
    int a=t(i),b=t((i+1)%n),c=t(j),e=t((j+1)%n);
    double gain=d(a,b)+d(c,e)-d(a,c)-d(b,e);
    if(gain<=threshold) {
      if(gain>0)++positive_marginal;else if(gain==0)++zero;else ++negative;
      if(cap>0) {signed_sample.push_back({double(i),double(j),gain});
        std::sort(signed_sample.begin(),signed_sample.end(),[](const auto&l,const auto&r){return std::abs(l[2])<std::abs(r[2]);});
        if(int(signed_sample.size())>cap)signed_sample.pop_back();}
      continue;
    }
    auto class_start=Clock::now();
    ++improving; gain_total+=gain;
    std::array<int,4> anchors{a,c,b,e};
    std::array<int,4> ranks{rnk(a,c),rnk(c,a),rnk(b,e),rnk(e,b)};
    std::array<int,4> prefixes{prefix_next[a],prefix_next[c],prefix_prev[b],prefix_prev[e]};
    std::vector<bool> members(p,false),scans(p,false);
    bool cm=false,cs=false;
    for(int z=0;z<p;++z) {
      for(int h=0;h<4;++h) {
        bool available=ranks[h]<=w(z,anchors[h]);
        members[z]=members[z]||available;
        scans[z]=scans[z]||(available && ranks[h]<=prefixes[h]);
        if(z==0&&in_cl[anchors[h]]) {cm=cm||available;cs=cs||(available&&ranks[h]<=prefixes[h]);}
      }
      if(members[z]) {++member_count[z];member_gain[z]+=gain;}
      if(scans[z]) {++scan_count[z];scan_gain[z]+=gain;max_scan_gain[z]=std::max(max_scan_gain[z],gain);}
    }
    if(cm) {++cl_member_count;cl_member_gain+=gain;}
    if(cs) {++cl_scan_count;cl_scan_gain+=gain;}
    float fg=(df(a,b)+df(c,e)-df(a,c))-df(b,e);
    Record rec{i,j,gain,fg,ranks,members,scans,cm,cs};
    if(int(sample.size())<cap)sample.push_back(rec);
    if(!members[0]) {
      max_missed=std::max(max_missed,gain);
      for(int u:anchors)node_gain[u]+=gain/4;
      double rel=gain/length;int bin=0;for(double cut:{1e-6,1e-5,1e-4,1e-3,1e-2,1e-1})if(rel>cut)++bin;
      ++miss_hist[bin];int minrank=*std::min_element(ranks.begin(),ranks.end());++rank_hist[minrank];
      if(cap>0) {top.push_back(rec);std::sort(top.begin(),top.end(),[](const Record&l,const Record&r){if(l.gain!=r.gain)return l.gain>r.gain;return std::tie(l.i,l.j)<std::tie(r.i,r.j);});if(int(top.size())>cap)top.pop_back();}
    }
    classification_seconds+=elapsed(class_start);
  }
  double enumeration=elapsed(loop);auto package=Clock::now();
  auto records=[&](const std::vector<Record>& records) {
    py::list out;
    for(const auto&r:records) {
      py::dict o;o["i"]=r.i;o["j"]=r.j;o["raw_gain_f64"]=r.gain;o["native_formula_gain_f32"]=r.native_gain;
      o["representation_ranks"]=r.ranks;o["candidate_membership"]=r.membership;o["potentially_evaluated_all_nodes"]=r.scan;
      o["candidate_available_but_pruned"]=r.membership[0]&&!r.scan[0];
      o["initial_checklist_candidate_available"]=r.checklist_member;o["initial_checklist_potentially_evaluated"]=r.checklist_scan;
      o["candidate_available_without_initial_checklist_anchor"]=r.membership[0]&&!r.checklist_member;
      o["actually_evaluated_native"]=py::none();o["evaluated_but_not_selected_native"]=py::none();o["actually_accepted_native"]=py::none();
      out.append(o);
    }return out;
  };
  py::dict out;out["n"]=n;out["tour_length_f64"]=length;out["robust_threshold"]=threshold;
  out["valid_moves"]=valid;out["robust_improving_moves"]=improving;out["positive_marginal_moves"]=positive_marginal;out["zero_gain_moves"]=zero;out["negative_gain_moves"]=negative;
  out["robust_gain_mass"]=gain_total;out["membership_count"]=member_count;out["membership_gain_mass"]=member_gain;
  out["scan_count"]=scan_count;out["scan_gain_mass"]=scan_gain;out["max_scan_gain"]=max_scan_gain;
  out["largest_membership_missed_gain"]=max_missed;out["node_missed_gain_mass"]=node_gain;
  out["missed_relative_gain_histogram"]=miss_hist;out["missed_min_representation_rank_histogram"]=rank_hist;
  out["top_membership_missed_moves"]=records(top);out["sample_robust_moves"]=records(sample);
  out["sample_nearest_zero_nonrobust_raw_signed_gains_i_j_gain"]=signed_sample;
  out["initial_checklist_membership_count"]=cl_member_count;out["initial_checklist_scan_count"]=cl_scan_count;
  out["initial_checklist_membership_gain_mass"]=cl_member_gain;out["initial_checklist_scan_gain_mass"]=cl_scan_gain;
  out["setup_classification_seconds"]=setup;out["enumeration_and_classification_seconds"]=enumeration;
  out["robust_move_classification_seconds"]=classification_seconds;
  out["enumeration_and_bounded_sample_seconds"]=enumeration-classification_seconds;
  out["packaging_seconds"]=elapsed(package);
  return out;
}
PYBIND11_MODULE(r3_diagnostic,m) {m.def("scan",&scan,py::arg("coords"),py::arg("tour"),py::arg("neighbors"),py::arg("widths"),py::arg("checklist"),py::arg("cap")=32);}
