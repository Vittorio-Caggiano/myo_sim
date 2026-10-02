// Bolt contact pipeline (collision_driver + collision_primitive + smooth_hunt_crossley), float64 C port.
// Geometry conventions follow Bolt: plane normal = local y axis, capsule axis = local z axis, size = (radius, half
// length, radius). The broadphase filters (plane / bounding sphere / AABB / OBB) are reproduced as released,
// including Bolt's geom_rbound and geom_aabb values, because they decide which contacts exist.
#ifndef MYO_CONTACT_H_
#define MYO_CONTACT_H_

#define MYO_GEOM_PLANE 0
#define MYO_GEOM_SPHERE 2
#define MYO_GEOM_CAPSULE 3
#define MYO_BOLT_MINVAL 1e-15

typedef struct {
  int body;                       // simulator body id (0 = world / Bolt ground)
  int type;
  double size[3], pos[3], quat[4]; // local pose in body frame, quat w x y z
  double friction[3];             // static, dynamic, viscous
  double stiffness, dissipation, transition;
  int priority;
  double aabb_center[3], aabb_size[3], rbound;
} MyoGeom;

typedef struct {
  double dist, pos[3], normal[3], curvature, stiffness, dissipation, transition, friction[3];
  int g1, g2;
} MyoContact;

// geom world poses from body poses: bpos (nbody*3), bquat (nbody*4, wxyz) -> gpos (ngeom*3), grot (ngeom*9)
void myo_geom_poses(int ngeom, const MyoGeom* g, const double* bpos, const double* bquat, double* gpos, double* grot);

// broadphase + narrowphase over the Bolt pair list; returns the number of contacts written (<= naconmax)
int myo_collide(int ngeom, const MyoGeom* g, const double* gpos, const double* grot, int npair, const int* pairs,
                int naconmax, int* ncollision, MyoContact* out);

// Hunt-Crossley normal force with Bolt's friction model. v1, v2: velocities of the contact point on body1 / body2.
// Returns 1 and the force on body2 (body1 receives -force) when active, 0 otherwise.
int myo_hc_force(const MyoContact* c, const double v1[3], const double v2[3], double force[3]);

#endif  // MYO_CONTACT_H_
