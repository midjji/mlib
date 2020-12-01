#include <sqlite_orm.h>
#include <iostream>
using namespace std;




struct Row{
    int uid;
    int frame_id;
    int imo_id;
    float bb0, bb1, bb2, bb3;
    float x,y,z;
};
std::ostream& operator<<(std::ostream& os, Row row){
    os<<"Row: {"<<row.uid<<" "<<row.frame_id<<" "<<row.imo_id<<" "<<row.bb0<<" "<<row.bb1<<"...}";
    return os;
}

using namespace sqlite_orm;
void create_db(){
    auto storage = make_storage("xxxgt.db.sqlite",
                                make_table("boundingboxes",
                                           make_column("id", &Row::uid, autoincrement(), primary_key()),
                                           make_column("frame_id", &Row::frame_id),
                                           make_column("imo_id", &Row::imo_id),
                                           make_column("bb0", &Row::bb0),
                                           make_column("bb1", &Row::bb1),
                                           make_column("bb2", &Row::bb2),
                                           make_column("bb3", &Row::bb3),
                                           make_column("x", &Row::x),
                                           make_column("y", &Row::y),
                                           make_column("z", &Row::z)));
    storage.sync_schema();
    Row row{-1, 1,2,3,4,5,6,7,8,9};

    auto insertedId = storage.insert(row);
    cout << "insertedId = " << insertedId << endl;      //  insertedId = 8
    row.uid = insertedId;


    // list all rows
    if(auto user = storage.get_pointer<Row>(insertedId))
    {
        cout<< *user<<endl;
    }
    else
    {
        cout << "no user with id " << insertedId << endl;
    }

    auto allUsers = storage.get_all<Row>();
    cout << "allUsers (" << allUsers.size() << "):" << endl;
    for(auto &user : allUsers) {
        cout << storage.dump(user) << endl; //  dump returns std::string with json-like style object info. For example: { id : '1', first_name : 'Jonh', last_name : 'Doe', birth_date : '664416000', image_url : 'https://cdn1.iconfinder.com/data/icons/man-icon-set/100/man_icon-21-512.png', type_id : '3' }
    }


    //  SELECT * FROM users WHERE id < 10
    auto idLesserThan10 = storage.get_all<Row>(where(c(&Row::uid) < 10));
    cout << "idLesserThan10 count = " << idLesserThan10.size() << endl;
    for(auto &user : idLesserThan10)
    {
        cout << storage.dump(user) << endl;
    }
}



int main(){
    //sudo apt-get install sqlite3-dev sqlite3 sqlitebrowser
    create_db();
    return 0;
}
