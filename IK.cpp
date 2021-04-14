#include "IK.h"
#include "FK.h"
#include "minivectorTemplate.h"
#include <Eigen/Dense>
#include <adolc/adolc.h>
#include <cassert>
#if defined(_WIN32) || defined(WIN32)
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#endif
#include <math.h>
using namespace std;

// CSCI 520 Computer Animation and Simulation
// Jernej Barbic and Yijing Li

namespace
{

	// Converts degrees to radians.
	template<typename real>
	inline real deg2rad(real deg) { return deg * M_PI / 180.0; }

	template<typename real>
	Mat3<real> Euler2Rotation(const real angle[3], RotateOrder order)
	{
		Mat3<real> RX = Mat3<real>::getElementRotationMatrix(0, deg2rad(angle[0]));
		Mat3<real> RY = Mat3<real>::getElementRotationMatrix(1, deg2rad(angle[1]));
		Mat3<real> RZ = Mat3<real>::getElementRotationMatrix(2, deg2rad(angle[2]));

		switch (order)
		{
		case RotateOrder::XYZ:
			return RZ * RY * RX;
		case RotateOrder::YZX:
			return RX * RZ * RY;
		case RotateOrder::ZXY:
			return RY * RX * RZ;
		case RotateOrder::XZY:
			return RY * RZ * RX;
		case RotateOrder::YXZ:
			return RZ * RX * RY;
		case RotateOrder::ZYX:
			return RX * RY * RZ;
		}
		assert(0);
	}

	// Performs forward kinematics, using the provided "fk" class.
	// This is the function whose Jacobian matrix will be computed using adolc.
	// numIKJoints and IKJointIDs specify which joints serve as handles for IK:
	//   IKJointIDs is an array of integers of length "numIKJoints"
	// Input: numIKJoints, IKJointIDs, fk, eulerAngles (of all joints)
	// Output: handlePositions (world-coordinate positions of all the IK joints; length is 3 * numIKJoints)
	template<typename real>
	void forwardKinematicsFunction(
		int numIKJoints, const int* IKJointIDs, const FK& fk,
		const std::vector<real>& eulerAngles, std::vector<real>& handlePositions)
	{
		// Students should implement this.
		// The implementation of this function is very similar to function computeLocalAndGlobalTransforms in the FK class.
		// The recommended approach is to first implement FK::computeLocalAndGlobalTransforms.
		// Then, implement the same algorithm into this function. To do so,
		// you can use fk.getJointUpdateOrder(), fk.getJointRestTranslation(), and fk.getJointRotateOrder() functions.
		// Also useful is the multiplyAffineTransform4ds function in minivectorTemplate.h .
		// It would be in principle possible to unify this "forwardKinematicsFunction" and FK::computeLocalAndGlobalTransforms(),
		// so that code is only written once. We considered this; but it is actually not easily doable.
		// If you find a good approach, feel free to document it in the README file, for extra credit.

		int fkNumJoints = fk.getNumJoints();

		vector<Mat3<real>> localRotations(fkNumJoints);
		vector<Vec3<real>> restLocalTranslations(fkNumJoints);

		for (int i = 0; i < fkNumJoints; i++)
		{
			// "RChild"
			Mat3<real> RChild = Euler2Rotation(&eulerAngles[i * 3], fk.getJointRotateOrder(i));

			// jointOrient
			Vec3d curJointOrient = fk.getJointOrient(i);
			real curJointOrientAngles[3] = { curJointOrient[0], curJointOrient[1], curJointOrient[2] };
			Mat3<real> RJointOrient = Euler2Rotation(curJointOrientAngles, RotateOrder::XYZ);

			localRotations[i] = RJointOrient * RChild;

			Vec3d curRestTranslation = fk.getJointRestTranslation(i);
			restLocalTranslations[i].set(real(curRestTranslation[0]), real(curRestTranslation[1]), real(curRestTranslation[2]));
		}

		vector<Mat3<real>> globalRotations(fkNumJoints);
		vector<Vec3<real>> globalTranslations(fkNumJoints);
		for (int i = 0; i < fkNumJoints; i++)
		{
			int curJointIndex = fk.getJointUpdateOrder(i);
			int curParentJointIndex = fk.getJointParent(curJointIndex);
			if (curParentJointIndex == -1) // root
			{
				globalRotations[curJointIndex] = localRotations[curJointIndex];
				globalTranslations[curJointIndex] = restLocalTranslations[curJointIndex];
			}
			else
			{
				multiplyAffineTransform4ds(
					globalRotations[curParentJointIndex], globalTranslations[curParentJointIndex], // parent global
					localRotations[curJointIndex], restLocalTranslations[curJointIndex], // current local
					globalRotations[curJointIndex], globalTranslations[curJointIndex]); // current global
			}
		}

		for (int i = 0; i < numIKJoints; i++)
		{
			int curIKJointIndex = IKJointIDs[i];
			handlePositions[3 * i + 0] = globalTranslations[curIKJointIndex][0];
			handlePositions[3 * i + 1] = globalTranslations[curIKJointIndex][1];
			handlePositions[3 * i + 2] = globalTranslations[curIKJointIndex][2];
		}

	}

} // end anonymous namespaces


void IK::switchIKMethod()
{
	useTikhonovRegularization = !useTikhonovRegularization;
	printf("current IK method: %s\n", (useTikhonovRegularization ? "Tikhonov regularization" : "PseudoInverse"));
}

void IK::switchSubStepIKOnOff()
{
	enableSubStepIK = !enableSubStepIK;
	printf("Sub-step IK %s.\n", (enableSubStepIK ? "Enabled" : "Disabled"));
}

void IK::performTikhonovRegularization(const Eigen::MatrixXd& J_EigenMat, const Eigen::VectorXd& deltaB_EigenVec, Eigen::VectorXd& deltaTheta_EigenVec)
{
	const double alpha = 0.01; // ? let me try 0.01 anyway

	Eigen::MatrixXd JT_EigenMat = J_EigenMat.transpose();

	int n = J_EigenMat.cols();

	// Identity's dimension: JT*J is (n by m)*(m by n) so (n by n)
	Eigen::MatrixXd I_EigenMat = Eigen::MatrixXd::Identity(n, n);

	Eigen::VectorXd rhs = JT_EigenMat * deltaB_EigenVec;

	// now we get everything needed to solve (JT*J + alpha*I)*deltaTheta = JT*deltaB
	Eigen::MatrixXd system_EigenMat = (JT_EigenMat * J_EigenMat) + (alpha * I_EigenMat);

	// the one we need to solve for. n*1 vec
	deltaTheta_EigenVec = system_EigenMat.ldlt().solve(rhs);
	assert(deltaTheta_EigenVec.rows() == n);
}

void IK::performPseudoInverse(const Eigen::MatrixXd& J_EigenMat, const Eigen::VectorXd& deltaB_EigenVec, Eigen::VectorXd& deltaTheta_EigenVec)
{
	Eigen::MatrixXd JT_EigenMat = J_EigenMat.transpose();

	int n = J_EigenMat.cols();

	// the pseudoinverse: JDagger
	Eigen::MatrixXd JDagger_EigenMat = JT_EigenMat * (J_EigenMat * JT_EigenMat).inverse();

	// then deltaTheta = JDagger * deltaB
	deltaTheta_EigenVec = JDagger_EigenMat * deltaB_EigenVec;
	assert(deltaTheta_EigenVec.rows() == n);
}

IK::IK(int numIKJoints, const int* IKJointIDs, FK* inputFK, int adolc_tagID)
{
	this->numIKJoints = numIKJoints;
	this->IKJointIDs = IKJointIDs;
	this->fk = inputFK;
	this->adolc_tagID = adolc_tagID;

	FKInputDim = fk->getNumJoints() * 3;
	FKOutputDim = numIKJoints * 3;

	train_adolc();
}

void IK::train_adolc()
{
	// Students should implement this.
	// Here, you should setup adol_c:
	//   Define adol_c inputs and outputs. 
	//   Use the "forwardKinematicsFunction" as the function that will be computed by adol_c.
	//   This will later make it possible for you to compute the gradient of this function in IK::doIK
	//   (in other words, compute the "Jacobian matrix" J).
	// See ADOLCExample.cpp .
	trace_on(adolc_tagID);

	vector<adouble> vecADolCInput(FKInputDim);
	for (int i = 0; i < FKInputDim; i++) {
		vecADolCInput[i] <<= 0.0;
	}
	vector<adouble> vecADolCOutput(FKOutputDim);

	::forwardKinematicsFunction(numIKJoints, IKJointIDs, *fk, vecADolCInput, vecADolCOutput);

	vector<double> vecRealOutput(FKOutputDim);
	for (int i = 0; i < FKOutputDim; i++) {
		vecADolCOutput[i] >>= vecRealOutput[i];
	}

	trace_off();
}

void IK::doIK(const Vec3d* targetHandlePositions, Vec3d* jointEulerAngles, const int maxIKIters, const double maxOneStepDistance)
{
	// You may find the following helpful:
	int numJoints = fk->getNumJoints(); // Note that is NOT the same as numIKJoints!

	// Students should implement this.
	// Use adolc to evalute the forwardKinematicsFunction and its gradient (Jacobian). It was trained in train_adolc().
	// Specifically, use ::function, and ::jacobian .
	// See ADOLCExample.cpp .

	// deltaB = m*1 vector: the change of handlePositions (m = 3*numIKJoints )// FKOutputDim // (targetHandlePositions - fk->currentHandlePositions)
	// deltaTheta = n*1 vector: the change of Euler angles (n = 3*fk->getNumJoints()) // FKInputDim
	// J & JT: computed by ::jacobian()
	// alpha: up to me
	// use ::function 
	// output should be written to jointEulerAngles

	int m = FKOutputDim, // for deltaB: m*1 vec, 3*numIKJoints
		n = FKInputDim;  // for deltaTheta: n*1 vec, this is the one we are solving for
	vector<double> deltaTheta(n);
	vector<double> deltaB(m);
	vector<double> curHandlePositions(m); // used for calc deltaB

	// now we get the current handle positions by ::function
	::function(adolc_tagID, m, n, jointEulerAngles->data(), curHandlePositions.data());

	vector<double> jacobianMatrix(m * n); // okay row-major
	vector<double*>  jacobianMatrixEachRow(m); // m rows 
	for (int i = 0; i < m; i++)
	{
		jacobianMatrixEachRow[i] = &jacobianMatrix[i * n];
	}
	// calc the jacobian matrix by ::jacobian
	::jacobian(adolc_tagID, m, n, jointEulerAngles->data(), jacobianMatrixEachRow.data());


	// deltaB
	Eigen::VectorXd deltaB_EigenVec(m);
	for (int i = 0; i < numIKJoints; i++)
	{
		deltaB_EigenVec(3 * i) = targetHandlePositions[i][0] - curHandlePositions[3 * i];
		deltaB_EigenVec(3 * i + 1) = targetHandlePositions[i][1] - curHandlePositions[3 * i + 1];
		deltaB_EigenVec(3 * i + 2) = targetHandlePositions[i][2] - curHandlePositions[3 * i + 2];
	}

	// the one we need to solve for. n*1 vec
	Eigen::VectorXd deltaTheta_EigenVec;

	int realIKIters = 1;

	if (enableSubStepIK)
	{
		// calculate the exact number of iterations needed this time
		double maxTotalDistance = 0;
		for (int i = 0; i < numIKJoints; i++)
		{
			Vec3d curTotalDeltaB(deltaB_EigenVec(3 * i), deltaB_EigenVec(3 * i + 1), deltaB_EigenVec(3 * i + 2));
			double curDistance = len(curTotalDeltaB);
			maxTotalDistance = max(maxTotalDistance, curDistance);
		}
		realIKIters = ceil(maxTotalDistance / maxOneStepDistance + 0.5);
		realIKIters = min(realIKIters, maxIKIters);
		printf("maxTotalDistance: %lf, maxOneStepDistance: %lf, realIKIters: %d.\n", maxTotalDistance, maxOneStepDistance, realIKIters);
	}

	// now perform sub-step IK
	for (int iter = 0; iter < realIKIters; iter++)
	{
		if (iter != 0)
		{
			// if it's the first iteration we can simply use the results above
			// if not, we need to re-calculate the handlePositions & deltaB

			// new current handle positions by ::function
			::function(adolc_tagID, m, n, jointEulerAngles->data(), curHandlePositions.data());

			// new jacobian matrix by ::jacobian
			::jacobian(adolc_tagID, m, n, jointEulerAngles->data(), jacobianMatrixEachRow.data());

			// new deltaB
			Eigen::VectorXd deltaB_EigenVec(m);
			for (int i = 0; i < numIKJoints; i++)
			{
				deltaB_EigenVec(3 * i) = targetHandlePositions[i][0] - curHandlePositions[3 * i];
				deltaB_EigenVec(3 * i + 1) = targetHandlePositions[i][1] - curHandlePositions[3 * i + 1];
				deltaB_EigenVec(3 * i + 2) = targetHandlePositions[i][2] - curHandlePositions[3 * i + 2];
			}
		}

		Eigen::MatrixXd J_EigenMat(m, n);
		for (int rowID = 0; rowID < m; rowID++)
		{
			for (int colID = 0; colID < n; colID++)
			{
				J_EigenMat(rowID, colID) = jacobianMatrix[n * rowID + colID];
			}
		}

		// we cannot bluntly use the deltaB above
		// because after each substep, the posDiff(Vec3)'s direction changes
		if (realIKIters > 1) { deltaB_EigenVec /= (double)(realIKIters - iter); }

		// Use it implement the Tikhonov IK method (or the pseudoinverse method for extra credit).
		// Note that at entry, "jointEulerAngles" contains the input Euler angles. 
		// Upon exit, jointEulerAngles should contain the new Euler angles.
		if (useTikhonovRegularization)
		{
			performTikhonovRegularization(J_EigenMat, deltaB_EigenVec, deltaTheta_EigenVec);
		}
		else
		{
			performPseudoInverse(J_EigenMat, deltaB_EigenVec, deltaTheta_EigenVec);
		}

		for (int i = 0; i < numJoints; i++)
		{
			jointEulerAngles[i] += Vec3d(deltaTheta_EigenVec(i * 3), deltaTheta_EigenVec(i * 3 + 1), deltaTheta_EigenVec(i * 3 + 2));
		}

	}

	/*
	if (useTikhonovRegularization) // Tikhonov regularization
	{
		const double alpha = 0.01; // ? let me try 0.01 anyway

		Eigen::MatrixXd J_EigenMat(m, n);
		for (int rowID = 0; rowID < m; rowID++)
		{
			for (int colID = 0; colID < n; colID++)
			{
				J_EigenMat(rowID, colID) = jacobianMatrix[n * rowID + colID];
			}
		}
		Eigen::MatrixXd JT_EigenMat(n, m);
		JT_EigenMat = J_EigenMat.transpose();

		// Identity's dimension: JT*J is (n by m)*(m by n) so (n by n)
		Eigen::MatrixXd I_EigenMat = Eigen::MatrixXd::Identity(n, n);

		Eigen::VectorXd rhs = JT_EigenMat * deltaB_EigenVec;

		// now we get everything needed to solve (JT*J + alpha*I)*deltaTheta = JT*deltaB
		Eigen::MatrixXd system_EigenMat = (JT_EigenMat * J_EigenMat) + (alpha * I_EigenMat);

		// the one we need to solve for. n*1 vec
		deltaTheta_EigenVec = system_EigenMat.ldlt().solve(rhs);
		assert(deltaTheta_EigenVec.rows() == n);
	}
	else // pseudoinverse
	{
		Eigen::MatrixXd J_EigenMat(m, n);
		for (int rowID = 0; rowID < m; rowID++)
		{
			for (int colID = 0; colID < n; colID++)
			{
				J_EigenMat(rowID, colID) = jacobianMatrix[n * rowID + colID];
			}
		}
		Eigen::MatrixXd JT_EigenMat(n, m);
		JT_EigenMat = J_EigenMat.transpose();

		// the pseudoinverse: JDagger
		Eigen::MatrixXd JDagger_EigenMat = JT_EigenMat * (J_EigenMat * JT_EigenMat).inverse();

		// then deltaTheta = JDagger * deltaB
		deltaTheta_EigenVec = JDagger_EigenMat * deltaB_EigenVec;
		assert(deltaTheta_EigenVec.rows() == n);
	}

	// output
	for (int i = 0; i < numJoints; i++)
	{
		jointEulerAngles[i] += Vec3d(deltaTheta_EigenVec(i * 3), deltaTheta_EigenVec(i * 3 + 1), deltaTheta_EigenVec(i * 3 + 2));
	}
	*/
}

