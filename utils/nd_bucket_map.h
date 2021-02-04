#pragma once
#include <mlib/utils/cvl/matrix.h>
#include <mlib/utils/cvl/matrix_adapter.h>
#include <vector>

namespace cvl {



/**
 * @brief The Fast2DQuery class
 *
 * For when you want to answer the question which element at position x is closest to the query position many times.
 * And know lots about the input data, the search data and building this index must be fast too.
 *
 * We seek the nearest neighbour within a maximum radius, and all points are in a range.
 *
 * \note
 * - ideally the query data should be uniformely distributed in the domain.
 *     You can transform query data to be closer to uniformely distributed by doing the kernel density and looking ad the cdf along each dim.
 *     That is good, but not ideal, another alternative is kmeans/voronoi but that requires finding, then adding, then adding multiple times.

 *
 * There should be a low number of elements per bucket, less than 32, or performance will suffer...
 *
 * Similar to a kd tree for seach, except faster and fixed.
 *
 * It needs the space span, x \in ()
 * and the number of buckets it should use per dimension.
 *
 */
class Fast2DQuery
{

public:
    struct Data {
        int index;
        Vector2d x;
            };

    Fast2DQuery(Vector2d minv,
            Vector2d maxv,
                Vector2i buckets);
    Vector2i position(Vector2d y);
    // a negative number means not found
    int find(Vector2d query, double max_radius);
    void add(Data data);

private:
    std::vector<Vector2i> query_locations(Vector2d y, double max_radius);
    /// lowest values
    Vector2d minv;
    /// highest values
    Vector2d maxv;


    /// bucket size x direction
    Vector2d delta;
    /// grid matrix adapter
    MatrixAdapter<std::vector<Data>> grid;
    /// grid matrix data
    std::vector<std::vector<Data>> griddata;

};

}
