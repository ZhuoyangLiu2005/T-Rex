from .paths import RobotModel, RobotType, URDFModel


class DexdGripperModel(RobotModel):
    @property
    def dexd_gripper(self) -> URDFModel:
        return URDFModel(self._type, self._name, "dexd_gripper")


class F5d6HandModel(RobotModel):
    @property
    def f5d6_right(self) -> URDFModel:
        return URDFModel(self._type, self._name, "f5d6_right")

    @property
    def f5d6_left(self) -> URDFModel:
        return URDFModel(self._type, self._name, "f5d6_left")


class DexsGripperModel(RobotModel):
    @property
    def dm_gripper(self) -> URDFModel:
        return URDFModel(self._type, self._name, "dm_gripper")


class Vega1pModel(RobotModel):
    @property
    def vega_1p(self) -> URDFModel:
        return URDFModel(self._type, self._name, "vega_1p")

    @property
    def vega_1p_f5d6(self) -> URDFModel:
        return URDFModel(self._type, self._name, "vega_1p_f5d6")

    @property
    def vega_1p_gripper(self) -> URDFModel:
        return URDFModel(self._type, self._name, "vega_1p_gripper")


class Vega1uModel(RobotModel):
    @property
    def vega_1u_f5d6(self) -> URDFModel:
        return URDFModel(self._type, self._name, "vega_1u_f5d6")

    @property
    def vega_1u_gripper(self) -> URDFModel:
        return URDFModel(self._type, self._name, "vega_1u_gripper")

    @property
    def vega_1u(self) -> URDFModel:
        return URDFModel(self._type, self._name, "vega_1u")


class Vega1Model(RobotModel):
    @property
    def vega(self) -> URDFModel:
        return URDFModel(self._type, self._name, "vega")

    @property
    def vega_1_f5d6(self) -> URDFModel:
        return URDFModel(self._type, self._name, "vega_1_f5d6")

    @property
    def vega_1_gripper(self) -> URDFModel:
        return URDFModel(self._type, self._name, "vega_1_gripper")

    @property
    def vega_1(self) -> URDFModel:
        return URDFModel(self._type, self._name, "vega_1")


class HandsType(RobotType):
    @property
    def dexd_gripper(self) -> DexdGripperModel:
        return DexdGripperModel("hands", "dexd_gripper")

    @property
    def f5d6_hand(self) -> F5d6HandModel:
        return F5d6HandModel("hands", "f5d6_hand")

    @property
    def dexs_gripper(self) -> DexsGripperModel:
        return DexsGripperModel("hands", "dexs_gripper")


class HumanoidType(RobotType):
    @property
    def vega_1p(self) -> Vega1pModel:
        return Vega1pModel("humanoid", "vega_1p")

    @property
    def vega_1u(self) -> Vega1uModel:
        return Vega1uModel("humanoid", "vega_1u")

    @property
    def vega_1(self) -> Vega1Model:
        return Vega1Model("humanoid", "vega_1")


hands = HandsType("hands")
humanoid = HumanoidType("humanoid")


def get_all_robot_dirs() -> list[RobotModel]:
    return []
