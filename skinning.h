#ifndef SKINNING_H
#define SKINNING_H

#include "transform4d.h"
#include <vector>
#include <string>

#include <math.h>

// A class to perform linear blend skinning on a triangle mesh.

// CSCI 520 Computer Animation and Simulation
// Jernej Barbic and Yijing Li

class Skinning
{
public: // util functions
	// switch current skinning mode in between LBS (linear-blend skinning) and DQS (dual-quaternion skinning)
	// the skinning mode is LBS by default.
	void switchSkinningMode();

protected:
	bool DQSMode = false;

public:
	// Load skinning data from a file.
	// numMeshVertices, restMeshVertexPositions: specifies the mesh vertices to be skinned
	// restMeshVertexPositions must be an array of length 3*numMeshVertices .
	// meshSkinningWeightsFilename: ASCII file in SparseMatrix format, giving the skinning weights.
	Skinning(int numMeshVertices, const double* restMeshVertexPositions, const std::string& meshSkinningWeightsFilename);

	// Main routine: Apply skinning to produce the new positions of the mesh vertices.
	// jointSkinTransforms is an array of transformations, one per joint. For each joint, we have: 
	// jointSkinTransform = globalTransform * globalRestTransform^{-1}
	// input: jointSkinTransforms
	// output: newMeshVertexPositions (length is 3*numMeshVertices)
	void applySkinning(const RigidTransform4d* jointSkinTransforms, const int numJoints, double* newMeshVertexPositions) const;


protected:
	static inline Vec4d Rotation2QuaternionVec4d(const Mat3d& rot);
	static inline Mat3d QuaternionVec4d2Rotation(const Vec4d& quat);
	static inline Vec4d QuaternionMultiply(const Vec4d& lhs, const Vec4d& rhs);
	static inline Vec4d GetQuaternionInverse(const Vec4d& quat);

protected:
	int numMeshVertices = 0;
	const double* restMeshVertexPositions = nullptr; // length of array is 3 x numMeshVertices

	// Number of joints that influence each vertex. This is constant for all vertices.
	int numJointsInfluencingEachVertex = 0;
	// The indices of joints that affect each mesh vertex.
	// Length is: numJointsInfluencingEachVertex * numMeshVertices.
	std::vector<int> meshSkinningJoints;
	// The skinning weights for each mesh vertex.
	// Length is numJointsInfluencingEachVertex * numMeshVertices.
	std::vector<double> meshSkinningWeights;
};


// ----------------- below are implementations of inline functions ---------------------

inline Vec4d Skinning::Rotation2QuaternionVec4d(const Mat3d& rot)
{
	// found from hw2, which follows the concepts of this paper
	// http://www.cs.cmu.edu/~baraff/pbm/pbm.html
	const double* R = rot.data();

	/*
	   Order of matrix elements is row-major:

	   (0,0) 0  (0,1) 1  (0,2) 2
	   (1,0) 3  (1,1) 4  (1,2) 5
	   (2,0) 6  (2,1) 7  (2,2) 8
	*/

	Vec4d q;
	double trace, u;
	trace = R[0] + R[4] + R[8];

	if (trace >= 0)
	{
		u = (double)sqrt(trace + 1);
		q[0] = (double)0.5 * u;
		u = (double)0.5 / u;
		q[1] = (R[7] - R[5]) * u;
		q[2] = (R[2] - R[6]) * u;
		q[3] = (R[3] - R[1]) * u;
	}
	else
	{
		int i = 0;
		if (R[4] > R[0])
			i = 1;

		if (R[8] > R[3 * i + i])
			i = 2;

		switch (i)
		{
		case 0:
			u = (double)sqrt((R[0] - (R[4] + R[8])) + 1);
			q[1] = 0.5f * u;
			u = 0.5f / u;
			q[2] = (R[3] + R[1]) * u;
			q[3] = (R[2] + R[6]) * u;
			q[0] = (R[7] - R[5]) * u;
			break;

		case 1:
			u = (double)sqrt((R[4] - (R[8] + R[0])) + 1);
			q[2] = 0.5f * u;
			u = 0.5f / u;
			q[3] = (R[7] + R[5]) * u;
			q[1] = (R[3] + R[1]) * u;
			q[0] = (R[2] - R[6]) * u;
			break;

		case 2:
			u = (double)sqrt((R[8] - (R[0] + R[4])) + 1);
			q[3] = 0.5f * u;

			u = 0.5f / u;
			q[1] = (R[2] + R[6]) * u;
			q[2] = (R[7] + R[5]) * u;
			q[0] = (R[3] - R[1]) * u;
			break;
		}
	}

	return q;
}

inline Mat3d Skinning::QuaternionVec4d2Rotation(const Vec4d& quat)
{
	/*
	* found from hw2, which follows the concepts of this paper
	* http://www.cs.cmu.edu/~baraff/pbm/pbm.html
	// Transforms the quaternion to the corresponding rotation matrix.
	// Quaternion is assumed to be a unit quaternion.
	// R is a 3x3 orthogonal matrix and will be returned in row-major order.
	template <typename real>
	inline void Quaternion<real>::Quaternion2Matrix(real* R) const
	{
		R[0] = 1 - 2 * y * y - 2 * z * z; R[1] = 2 * x * y - 2 * s * z;     R[2] = 2 * x * z + 2 * s * y;
		R[3] = 2 * x * y + 2 * s * z;     R[4] = 1 - 2 * x * x - 2 * z * z; R[5] = 2 * y * z - 2 * s * x;
		R[6] = 2 * x * z - 2 * s * y;     R[7] = 2 * y * z + 2 * s * x;     R[8] = 1 - 2 * x * x - 2 * y * y;
	}
	*/
	double R[9];
	double s = quat[0], x = quat[1], y = quat[2], z = quat[3];
	R[0] = 1 - 2 * y * y - 2 * z * z; R[1] = 2 * x * y - 2 * s * z;     R[2] = 2 * x * z + 2 * s * y;
	R[3] = 2 * x * y + 2 * s * z;     R[4] = 1 - 2 * x * x - 2 * z * z; R[5] = 2 * y * z - 2 * s * x;
	R[6] = 2 * x * z - 2 * s * y;     R[7] = 2 * y * z + 2 * s * x;     R[8] = 1 - 2 * x * x - 2 * y * y;

	return Mat3d(R);
}

inline Vec4d Skinning::QuaternionMultiply(const Vec4d& lhs, const Vec4d& rhs)
{
	/* quaternion multiplication, found from hw2
	template <typename real>
	inline Quaternion<real> Quaternion<real>::operator* (const Quaternion<real> q2) const
	{
		Quaternion<real> w(
			s * q2.s - x * q2.x - y * q2.y - z * q2.z,
			s * q2.x + q2.s * x + y * q2.z - q2.y * z,
			s * q2.y + q2.s * y + q2.x * z - x * q2.z,
			s * q2.z + q2.s * z + x * q2.y - q2.x * y);

		return w;
	}
	*/
	return Vec4d(
		lhs[0] * rhs[0] - lhs[1] * rhs[1] - lhs[2] * rhs[2] - lhs[3] * rhs[3],
		lhs[0] * rhs[1] + rhs[0] * lhs[1] + lhs[2] * rhs[3] - rhs[2] * lhs[3],
		lhs[0] * rhs[2] + rhs[0] * lhs[2] + rhs[1] * lhs[3] - lhs[1] * rhs[3],
		lhs[0] * rhs[3] + rhs[0] * lhs[3] + lhs[1] * rhs[2] - rhs[1] * lhs[2]
	);

}

inline Vec4d Skinning::GetQuaternionInverse(const Vec4d& quaternion)
{
	Vec4d conjugate(quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]);
	double sumSqr = len2(quaternion);
	return conjugate / sumSqr;
}


#endif