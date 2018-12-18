#include <cmath>
#include <lgr_when.hpp>
#include <limits>

namespace lgr {

void When::out_of_line_virtual_method() {}

struct TimePeriodic : public When {
  double period;
  TimePeriodic(double period_in) : period(period_in) {}
  void out_of_line_virtual_method() override;
  double next_event(double time) override final {
    using LL = long long;
    auto quotient = LL(std::floor(time / period));
    while (quotient * period > time) --quotient;
    while (quotient * period <= time) ++quotient;
    auto out = quotient * period;
    return out;
  }
  bool active(double prev_time, double time) override final {
    if (prev_time == time) {
      using LL = long long;
      auto quotient = LL(std::floor(time / period));
      return time == (quotient * period);
    }
    return next_event(prev_time) <= time;
  }
};

void TimePeriodic::out_of_line_virtual_method() {}

struct TimeRange : public When {
  double start;
  double end;
  TimeRange(double start_in, double end_in) : start(start_in), end(end_in) {}
  void out_of_line_virtual_method() override;
  double next_event(double time) override final {
    if (time < start) return start;
    if (time < end) return end;
    return std::numeric_limits<double>::max();
  }
  bool active(double, double time) override final {
    return (start <= time) && (time < end);
  }
};

void TimeRange::out_of_line_virtual_method() {}

struct AtTime : public When {
  double point;
  AtTime(double point_in) : point(point_in) {}
  void out_of_line_virtual_method() override;
  double next_event(double time) override final {
    if (time < point) return point;
    return std::numeric_limits<double>::max();
  }
  bool active(double prev_time, double time) override final {
    if (prev_time == time) return point == time;
    return (prev_time < point) && (point <= time);
  }
};

void AtTime::out_of_line_virtual_method() {}

struct Always : public When {
  void out_of_line_virtual_method() override;
  double next_event(double) override final {
    return std::numeric_limits<double>::max();
  }
  bool active(double, double) override final { return true; }
};

void Always::out_of_line_virtual_method() {}

struct Never : public When {
  void out_of_line_virtual_method() override;
  double next_event(double) override final {
    return std::numeric_limits<double>::max();
  }
  bool active(double, double) override final { return false; }
};

void Never::out_of_line_virtual_method() {}

When* time_periodic(double period) { return new TimePeriodic(period); }
When* time_range(double start, double end) { return new TimeRange(start, end); }
When* at_time(double point) { return new AtTime(point); }
When* always() { return new Always(); }
When* never() { return new Never(); }

When* setup_when(Omega_h::InputMap& pl) {
  if (pl.is<double>("time period")) {
    return time_periodic(pl.get<double>("time period"));
  }
  if (pl.is<double>("start time") && pl.is<double>("end time")) {
    return time_range(pl.get<double>("start time"), pl.get<double>("end time"));
  }
  if (pl.is<double>("at time")) {
    return at_time(pl.get<double>("at time"));
  }
  return always();
}

}  // namespace lgr
