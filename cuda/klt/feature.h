#pragma once

#include <cstdint>
#include <string>
#include <iostream>

namespace klt {


/*
 * Feature structure.
 */
struct SCUDAKLTFeature_t
{
    SCUDAKLTFeature_t()=default;
    SCUDAKLTFeature_t(float prev_u, float prev_v,
                      float u, float v,
                      float scale):
        prev_col_(prev_u), prev_row_(prev_v),
        current_col_(u), current_row_(v),
        scale(scale),
        status_i(0), residuum_f(0), toss(0){}



    /// Feature position in previous image.
    float  prev_col_=-1;
    float  prev_row_=-1;

    /// Feature position in current image.
    float  current_col_=-1;
    float  current_row_=-1;
    float  scale=1;

    /// Feature status. If 0, feature is tracked, otherwise error code.
    int32_t    status_i=10; // if not zero when sent to gpu, it skips it entierly

    /// Residuum on last pyramid level. (SAD)
    float  residuum_f=-1;
    int32_t toss=0; // force 32bit alignment

};
static_assert(sizeof(SCUDAKLTFeature_t)==32," must be size 32" );







/// Feature structure.
struct SFeature_t
{
    enum State{ FREE, LOST, FOUND, TRACKED};
    void clear();

    std::string str() const;
    SCUDAKLTFeature_t klt_feature(bool use_prediction) const;
    void update(const SCUDAKLTFeature_t& cuda_feature, double max_resid);




    SFeature_t()=default;
    SFeature_t(float row, float col,float pred_row=-1,float pred_col=-1);

    static SFeature_t undefined(){
        SFeature_t f;
        f.state=FREE;
        f.row_=0;
        f.col_=0;
        f.predrow_=-1;
        f.predcol_=-1;
        return f;
    }

    float row() const;
    float col() const;
    float previous_row() const;
    float previous_col() const;
    bool sensible_prediction() const;
    float predicted_row() const;
    float predicted_col() const;
    template<class T> T rc() const{return T(row(),col());}
    template<class T> T previous_rc() const{return T(previous_row(),previous_col());}
    template<class T> T predicted_rc() const {return T(predicted_row(),predicted_col());}



    void set(float row,
             float col,
             float predict_row=-1,
             float predict_col=-1);
    template<class RowCol>
    void set(RowCol y,
             RowCol yp)
    {
        set(y[0],y[1],yp[0],yp[1]);
    }



    bool tracked() const;
    bool found()   const;
    bool lost()    const;
    bool free()    const;

    bool is_lost_or_undefined() const;
    bool valid() const;
    bool tracked_or_found() const;






    /**
     * \brief  Age (nr of frames) of the feature.
     *
     * A FOUNDfeature has an age of 1.
     */
    int           age_ui=1;
    float         col_=-1;
    float         row_=-1;

    /// Last image position. This is only valid if state is TRACKED!
    float         lastcol_=-1;

    /// Last image position. This is only valid if state is TRACKED!
    float         lastrow_=-1;

    /**
     * \brief  Predicted image position in horizontal direction in next frame.
     *
     * If predcol_ is <= 0, these values are ignored.
     */
    float         predcol_=-1;

    /**
     * \brief  Predicted image position in vertical direction in next frame.
     *
     * If predcol_ is <= 0, these values are ignored.
     */
    float         predrow_=-1;
    float         residual=0;
    int       result=0;
private:
    State  state=FREE;
};

}
