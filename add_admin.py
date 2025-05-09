from app import app, db
from app import User
from werkzeug.security import generate_password_hash

def add_admin():
    with app.app_context():
        # Define the admin user details
        admin_username = "admin101"
        admin_email = "admin@gmail.com"
        admin_password = "admin101"
        admin_role = "admin"
        
        # Check if the admin user already exists
        if User.query.filter_by(email=admin_email).first():
            print("Admin user already exists.")
            return
        
        # Create a new user with the admin role
        hashed_password = generate_password_hash(admin_password)
        admin_user = User(username=admin_username, email=admin_email, password=hashed_password, role=admin_role)
        
        # Add and commit the new admin user to the database
        db.session.add(admin_user)
        db.session.commit()
        print("Admin user created successfully.")

if __name__ == "__main__":
    add_admin()