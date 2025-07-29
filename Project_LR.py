import sqlite3

conn = sqlite3.connect("Database.db")
cursor = conn.cursor()


cursor.execute("""
    CREATE TABLE IF NOT EXISTS Housing_Price(
               Area  INTEGER not null,
               Bedrooms INTEGER not null,
               Bathrooms INTEGER not null,
               Stories INTEGER not null,
               Mainroad INTEGER not null,
               Guestroom INTEGER not null,
               Basement INTEGER not null,
               Airconditioning INTEGER not null,
               Parking INTEGER not null,
               Prefarea INTEGER not null,
               Furnishingstatus INTEGER not null
               )
""")


def Insert_Values(*args):
    cursor.execute("INSERT into Housing_Price (Area,Bedrooms,Bathrooms,Stories,Mainroad,Guestroom,Basement,Airconditioning,Parking,Prefarea,Furnishingstatus) values (?,?,?,?,?,?,?,?,?,?,?)",args)
    conn.commit()

def ShowData():
    Data = cursor.execute("Select * from Housing_Price").fetchall()
    for data in Data:
        print(data)



def main():

    area = int(input("Enter value of Area"))
    bedroom = int(input("Enter value of bd"))
    bathroom = int(input("Enter value of bth"))
    stories = int(input("Enter value of sto"))
    mainroad = int(input("Enter value of main"))
    guestroom = int(input("Enter value of guest"))
    basement = int(input("Enter value of bst"))
    airconditioning = int(input("Enter value of air"))
    parking = int(input("Enter value of prk"))
    prearea = int(input("Enter value of prfar"))
    status = int(input("Enter value of stas"))
    Insert_Values(area,bedroom,bathroom,stories,mainroad,guestroom,basement,airconditioning,parking,prearea,status)
    
    Show = input("You Want to See Data: ?")
    if Show.lower()=="yes":
        ShowData()
    else:
       exit()
    



if __name__ == "__main__":
    main()