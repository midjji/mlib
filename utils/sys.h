#pragma once
/**
 * Functionality related to common system tasks
 */
#include <string>
namespace mlib{
/**
 * System calls are notoriously problematic,
 * and should be avoided whenever possible,
 * I use this when I port older code though
 * but this one atleast informs you why it failed in a better way
 */
int saferSystemCall(std::string cmd);

}// end namespace mlib
