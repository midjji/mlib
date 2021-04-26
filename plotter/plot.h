#pragma once
#include <string>
#include <vector>
#include <map>
namespace cvl {


/**
 * @brief plot
 * @param xs
 * @param ys
 * @param title
 * \notes
 * - Thread Safe
 * - Initializes automatically on first request
 * - stops running once all windows have been closed.
 * - call twice with the same name and they show up in the
 *      same window with different colors, unless you closed it inbetween
 * - Implicitly internally runs a QApplication in its own thread
 * - Conflict with opencv? No, mtgui has trouble with blocking though
 * - Conflict with anything else, probably not.
 */
void plot(const std::vector<double>& xs,
          const std::vector<double>& ys,
          std::string title="untitled plot", std::string label="unnamed graph");
void plot(const std::vector<double>& ys,
          std::string title="untitled plot", std::string label="unnamed graph");
void plot(const std::vector<double>& xs,
          const std::map<std::string, std::vector<double>>& yss,
          std::string title="unnamed window");
void clear_plot(std::string title="unnamed window");
void initialize_plotter();






}
