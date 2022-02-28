#include <sstream>
#include "klt/feature.h"
#include <cmath>
namespace klt {


void SFeature_t::clear(){
    *this=SFeature_t();
}
std::string SFeature_t::str() const{
    std::stringstream os;
    switch(state)
    {
    case FREE:  os<<"state: UNDEFINED";   break;
    case LOST:       os<<"state: LOST";        break;
    case TRACKED:    os<<"state: TRACKED";     break;
    case FOUND:      os<<"state: FOUND";       break;
    }
    os<<"\n";
    os<<"age:       "<<age_ui<<"\n";
    os<<"(row,col):       ("<<row()<<", "<<col()<<")\n";
    os<<"last (row,col):  ("<<previous_row()<<", "<<previous_col()<<")\n";
    os<<"pred row, col: "<<predicted_row()<<", " <<predicted_col()<<"\n";
    os<<"residual: "<<residual<<"\n";
    os<<"result: "<<result<<"\n";
    return os.str();
}
SFeature_t::SFeature_t(float row, float col,float pred_row,float pred_col):
    state(FOUND),row_(row),col_(col),predrow_(pred_row),predcol_(pred_col){}
SCUDAKLTFeature_t SFeature_t::klt_feature(bool use_prediction) const
{
    if(is_lost_or_undefined()) return SCUDAKLTFeature_t();/// a dont track feature

    SCUDAKLTFeature_t cuda_feature;
    cuda_feature.prev_col_    = col_;
    cuda_feature.prev_row_    = row_;
    cuda_feature.residuum_f = 0.0;
    cuda_feature.status_i   = 0; // to track!
    cuda_feature.current_col_ =  col_;
    cuda_feature.current_row_ = row_;

    if ( sensible_prediction()  && use_prediction)
    {
        cuda_feature.current_col_ = predcol_;
        cuda_feature.current_row_ = predrow_;
    }
    return cuda_feature;
}
void SFeature_t::update(const SCUDAKLTFeature_t& cuda_feature, double max_resid)
{
    // default to having lost it
    state        = LOST;
    result     = cuda_feature.status_i;
    if(cuda_feature.status_i  != 0 ) return;
    if((max_resid>0)&&cuda_feature.residuum_f>max_resid) return;
    state        = TRACKED;
    lastcol_      = col_;
    lastrow_      = row_;

    col_          = cuda_feature.current_col_;
    row_          = cuda_feature.current_row_;

    residual   = cuda_feature.residuum_f;
    ++age_ui;
}
float SFeature_t::row() const{return row_;} // row was v
float SFeature_t::col() const{return col_;} // col was u
float SFeature_t::previous_row() const{return lastrow_;}
float SFeature_t::previous_col() const{return lastcol_;}
bool SFeature_t::sensible_prediction() const {
    return predicted_col()>0 && predicted_row()>0;
}
float SFeature_t::predicted_row() const{return predrow_;}
float SFeature_t::predicted_col() const{return predcol_;}
void SFeature_t::set(float row,
                     float col,
                     float predict_row,
                     float predict_col){
    row_=row;
    col_=col;
    predrow_=predict_row;
    predcol_=predict_col;
    state=FOUND;
    age_ui=1;
}
bool SFeature_t::is_lost_or_undefined()   const{return free()||lost();    }
bool SFeature_t::tracked_or_found()       const{return tracked()||found();}
bool SFeature_t::valid()                  const{return tracked()||found();}
bool SFeature_t::normal()                  const{
    double v=row() + col()+ predicted_col()+predicted_row();
    if(std::isnan(v)) return false;
    if(std::isinf(v)) return false;
    return true;
}

bool SFeature_t::found()                  const{return state==FOUND;}
bool SFeature_t::lost()                   const{return state==LOST;}
bool SFeature_t::free()                   const {return state==FREE;}
bool SFeature_t::tracked()                const{return state==TRACKED;}

}
