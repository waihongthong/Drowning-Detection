import 'package:flutter/material.dart';
import 'editProfileScreen.dart';

class ProfileScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[50],
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        title: Text(
          'Profile',
          style: TextStyle(
            color: Colors.blue,
            fontWeight: FontWeight.bold,
          ),
        ),
        leading: IconButton(
          icon: Icon(Icons.arrow_back, color: Colors.blue),
          onPressed: () {},
        ),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          child: Column(
            children: [
              // Profile header
              Container(
                padding: EdgeInsets.symmetric(vertical: 24, horizontal: 16),
                color: Colors.white,
                child: Column(
                  children: [
                    // Profile image with edit button
                    Stack(
                      children: [
                        CircleAvatar(
                          radius: 40,
                          backgroundImage:
                              AssetImage('assets/profile_placeholder.png'),
                          // Use a placeholder image or NetworkImage if you have a URL
                        ),
                        Positioned(
                          bottom: 0,
                          right: 0,
                          child: Container(
                            padding: EdgeInsets.all(4),
                            decoration: BoxDecoration(
                              color: Colors.blue,
                              shape: BoxShape.circle,
                            ),
                            child: Icon(
                              Icons.edit,
                              color: Colors.white,
                              size: 16,
                            ),
                          ),
                        ),
                      ],
                    ),
                    SizedBox(height: 12),
                    // Name
                    Text(
                      'John Doe',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              ),

              SizedBox(height: 16),

              // Menu options
              Container(
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(8),
                ),
                margin: EdgeInsets.symmetric(horizontal: 16),
                child: Column(
                  children: [
                    _buildMenuOption(
                      context,
                      icon: Icons.person,
                      title: 'Profile',
                      iconColor: Colors.blue[600]!,
                      onTap: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                              builder: (context) => EditProfileScreen()),
                        );
                      },
                    ),
                    _divider(),
                    _buildMenuOption(
                      context,
                      icon: Icons.favorite,
                      title: 'Favorite',
                      iconColor: Colors.red[400]!,
                      onTap: () {},
                    ),
                    _divider(),
                    _buildMenuOption(
                      context,
                      icon: Icons.payment,
                      title: 'Payment Method',
                      iconColor: Colors.indigo[400]!,
                      onTap: () {},
                    ),
                    _divider(),
                    _buildMenuOption(
                      context,
                      icon: Icons.lock,
                      title: 'Privacy Policy',
                      iconColor: Colors.orange[700]!,
                      onTap: () {},
                    ),
                    _divider(),
                    _buildMenuOption(
                      context,
                      icon: Icons.settings,
                      title: 'Settings',
                      iconColor: Colors.grey[700]!,
                      onTap: () {},
                    ),
                    _divider(),
                    _buildMenuOption(
                      context,
                      icon: Icons.help,
                      title: 'Help',
                      iconColor: Colors.purple[400]!,
                      onTap: () {},
                    ),
                    _divider(),
                    _buildMenuOption(
                      context,
                      icon: Icons.logout,
                      title: 'Logout',
                      iconColor: Colors.blue[400]!,
                      onTap: () {},
                    ),
                  ],
                ),
              ),

              SizedBox(height: 16),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildMenuOption(
    BuildContext context, {
    required IconData icon,
    required String title,
    required Color iconColor,
    required VoidCallback onTap,
  }) {
    return InkWell(
      onTap: onTap,
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 16),
        child: Row(
          children: [
            Container(
              padding: EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: iconColor.withOpacity(0.1),
                shape: BoxShape.circle,
              ),
              child: Icon(
                icon,
                color: iconColor,
                size: 20,
              ),
            ),
            SizedBox(width: 16),
            Text(
              title,
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w500,
              ),
            ),
            Spacer(),
            Icon(
              Icons.arrow_forward_ios,
              color: Colors.grey,
              size: 16,
            ),
          ],
        ),
      ),
    );
  }

  Widget _divider() {
    return Divider(
      height: 1,
      thickness: 0.5,
      indent: 56,
      endIndent: 16,
    );
  }
}
