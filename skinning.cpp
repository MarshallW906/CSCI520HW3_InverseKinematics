#include "skinning.h"
#include "vec3d.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <fstream>
using namespace std;

// CSCI 520 Computer Animation and Simulation
// Jernej Barbic and Yijing Li

void Skinning::switchSkinningMode()
{
	DQSMode = !DQSMode;
	printf("current skinning mode: %s\n", (DQSMode ? "Dual-Quaternion" : "Linear-Blend"));
}


Skinning::Skinning(int numMeshVertices, const double* restMeshVertexPositions,
	const std::string& meshSkinningWeightsFilename)
{
	this->numMeshVertices = numMeshVertices;
	this->restMeshVertexPositions = restMeshVertexPositions;

	cout << "Loading skinning weights..." << endl;
	ifstream fin(meshSkinningWeightsFilename.c_str());
	assert(fin);
	int numWeightMatrixRows = 0, numWeightMatrixCols = 0;
	fin >> numWeightMatrixRows >> numWeightMatrixCols;
	assert(fin.fail() == false);
	assert(numWeightMatrixRows == numMeshVertices);
	int numJoints = numWeightMatrixCols;

	vector<vector<int>> weightMatrixColumnIndices(numWeightMatrixRows);
	vector<vector<double>> weightMatrixEntries(numWeightMatrixRows);
	fin >> ws;
	while (fin.eof() == false)
	{
		int rowID = 0, colID = 0;
		double w = 0.0;
		fin >> rowID >> colID >> w;
		weightMatrixColumnIndices[rowID].push_back(colID);
		weightMatrixEntries[rowID].push_back(w);
		assert(fin.fail() == false);
		fin >> ws;
	}
	fin.close();

	// Build skinning joints and weights.
	numJointsInfluencingEachVertex = 0;
	for (int i = 0; i < numMeshVertices; i++)
		numJointsInfluencingEachVertex = std::max(numJointsInfluencingEachVertex, (int)weightMatrixEntries[i].size());
	assert(numJointsInfluencingEachVertex >= 2);

	// Copy skinning weights from SparseMatrix into meshSkinningJoints and meshSkinningWeights.
	meshSkinningJoints.assign(numJointsInfluencingEachVertex * numMeshVertices, 0);
	meshSkinningWeights.assign(numJointsInfluencingEachVertex * numMeshVertices, 0.0);
	for (int vtxID = 0; vtxID < numMeshVertices; vtxID++)
	{
		vector<pair<double, int>> sortBuffer(numJointsInfluencingEachVertex);
		for (size_t j = 0; j < weightMatrixEntries[vtxID].size(); j++)
		{
			int frameID = weightMatrixColumnIndices[vtxID][j];
			double weight = weightMatrixEntries[vtxID][j];
			sortBuffer[j] = make_pair(weight, frameID);
		}
		sortBuffer.resize(weightMatrixEntries[vtxID].size());
		assert(sortBuffer.size() > 0);
		sort(sortBuffer.rbegin(), sortBuffer.rend()); // sort in descending order using reverse_iterators
		for (size_t i = 0; i < sortBuffer.size(); i++)
		{
			meshSkinningJoints[vtxID * numJointsInfluencingEachVertex + i] = sortBuffer[i].second;
			meshSkinningWeights[vtxID * numJointsInfluencingEachVertex + i] = sortBuffer[i].first;
		}

		// Note: When the number of joints used on this vertex is smaller than numJointsInfluencingEachVertex,
		// the remaining empty entries are initialized to zero due to vector::assign(XX, 0.0) .
	}
}

void Skinning::applySkinning(const RigidTransform4d* jointSkinTransforms, const int numJoints, double* newMeshVertexPositions) const
{
	// Students should implement this

	// here we cannot simply use sizeof(newMeshVertexPositions)
	// because when it's passed into this function it degenerates to a double* and it has no idea what the array size is
	memset(newMeshVertexPositions, 0, (3 * numMeshVertices * sizeof(double)));

	if (!DQSMode) { // LBS
		for (int i = 0; i < numMeshVertices; i++)
		{
			Vec3d restMeshVertexPosVec(&restMeshVertexPositions[3 * i]); // the rest pos of the i-th point
			for (int j = 0; j < numJointsInfluencingEachVertex; j++)
			{
				int curIndex = i * numJointsInfluencingEachVertex + j;
				int curJointIndex = meshSkinningJoints[curIndex];
				double curWeight = meshSkinningWeights[curIndex];

				Vec3d newWeightedPos = curWeight * jointSkinTransforms[curJointIndex].transformPoint(restMeshVertexPosVec);
				newMeshVertexPositions[3 * i + 0] += newWeightedPos[0];
				newMeshVertexPositions[3 * i + 1] += newWeightedPos[1];
				newMeshVertexPositions[3 * i + 2] += newWeightedPos[2];
			}
		}
	}
	else // DQS: Dual Quaternion Skinning
	{
		// for each jointSkinTransforms[i], we do:
		// 1. convert rotation to quaternion q0 (do we need to normalize it? I think it will be already normalized
		// 2. getTranslation(), then calc q1 = 0.5 * translation * q_0
		//    where we form a t_quat = (0, tx, ty, tz) and then apply quaternion product rules
		std::vector<Vec4d> q0s(numJoints), q1s(numJoints);

		for (int i = 0; i < numJoints; i++)
		{
			Mat3d curRotMat = jointSkinTransforms[i].getRotation();
			Vec3d curTranslation = jointSkinTransforms[i].getTranslation();

			q0s[i] = Skinning::Rotation2QuaternionVec4d(curRotMat); // do we need to force normalize it after the conversion?
			//q0s[i].normalize();
			Vec4d curTQuat(0, curTranslation[0], curTranslation[1], curTranslation[2]);
			q1s[i] = 0.5 * Skinning::QuaternionMultiply(curTQuat, q0s[i]);
		}

		// 3. for each MeshVertex
		// blend the (numJointsInfluencingEachVertex) 'q0's together and do the same on the 'q1's by applying the weights
		// when applying the weights, simply multiply the weight by each component
		std::vector<Vec4d> q0Blended(numMeshVertices), q1Blended(numMeshVertices);
		for (int i = 0; i < numMeshVertices; i++)
		{
			q0Blended[i].set(0);
			q1Blended[i].set(0);

			for (int j = 0; j < numJointsInfluencingEachVertex; j++)
			{
				int curIndex = i * numJointsInfluencingEachVertex + j;
				int curJointIndex = meshSkinningJoints[curIndex];
				double curWeight = meshSkinningWeights[curIndex];

				double sign = 1;
				if (q0s[curJointIndex][0] < 0) { sign = -1; }

				q0Blended[i] += curWeight * sign * q0s[curJointIndex];
				q1Blended[i] += curWeight * sign * q1s[curJointIndex];
			}

			// normalize both by q0Blended.length
			double q0LengthInv = 1 / len(q0Blended[i]);
			q0Blended[i] *= q0LengthInv;
			q1Blended[i] *= q0LengthInv;
		}
		

		// for each mesh vertex, use the associated q0Blended & q1Blended:
		// 4. retrive t by unpacking the blended q1
		//	  t = 2 * q1 * q0.inv() // quat.inv() == quat.conj() / (q0*q0 + q1*q1 + q2*q2 + q3*q3)^2
		//	  then we extract the vector part, which will be the translation
		// 5. also convert the blended q0 back to rotation matrix
		// 6. form a new RigidTransform4d by the rotation matrix & translation, and call RigidTransform4d::transformPoint()

		for (int i = 0; i < numMeshVertices; i++)
		{
			Mat3d curRotation = Skinning::QuaternionVec4d2Rotation(q0Blended[i]);
			Vec4d unpackedTQuat = 2 * Skinning::QuaternionMultiply(q1Blended[i], Skinning::GetQuaternionInverse(q0Blended[i]));
			// extract the vector part, the scalar part will always be zero
			Vec3d curTranslation(unpackedTQuat[1], unpackedTQuat[2], unpackedTQuat[3]);
			RigidTransform4d DQSTransform(curRotation, curTranslation);

			Vec3d restMeshVertexPosVec(&restMeshVertexPositions[3 * i]); // the rest pos of the i-th point
			Vec3d DQSPos = DQSTransform.transformPoint(restMeshVertexPosVec);
			newMeshVertexPositions[3 * i + 0] += DQSPos[0];
			newMeshVertexPositions[3 * i + 1] += DQSPos[1];
			newMeshVertexPositions[3 * i + 2] += DQSPos[2];
		}
	}
}
