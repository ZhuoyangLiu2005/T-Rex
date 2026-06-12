from .paths import RobotModel, RobotType, URDFModel


class Vega1Model(RobotModel):
    @property
    def vega_1_gripper(self) -> URDFModel:
        return URDFModel(self._type, self._name, "vega_1_gripper")

    @property
    def vega_1(self) -> URDFModel:
        return URDFModel(self._type, self._name, "vega_1")

    @property
    def vega_1_f5d6(self) -> URDFModel:
        return URDFModel(self._type, self._name, "vega_1_f5d6")

    @property
    def vega(self) -> URDFModel:
        return URDFModel(self._type, self._name, "vega")


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


class DexdGripperModel(RobotModel):
    @property
    def dexd_gripper(self) -> URDFModel:
        return URDFModel(self._type, self._name, "dexd_gripper")


class DexsGripperModel(RobotModel):
    @property
    def dm_gripper(self) -> URDFModel:
        return URDFModel(self._type, self._name, "dm_gripper")


class F5d6HandModel(RobotModel):
    @property
    def f5d6_left(self) -> URDFModel:
        return URDFModel(self._type, self._name, "f5d6_left")

    @property
    def f5d6_right(self) -> URDFModel:
        return URDFModel(self._type, self._name, "f5d6_right")


class HumanoidType(RobotType):
    @property
    def vega_1(self) -> Vega1Model:
        return Vega1Model("humanoid", "vega_1")

    @property
    def vega_1p(self) -> Vega1pModel:
        return Vega1pModel("humanoid", "vega_1p")

    @property
    def vega_1u(self) -> Vega1uModel:
        return Vega1uModel("humanoid", "vega_1u")


class HandsType(RobotType):
    @property
    def dexd_gripper(self) -> DexdGripperModel:
        return DexdGripperModel("hands", "dexd_gripper")

    @property
    def dexs_gripper(self) -> DexsGripperModel:
        return DexsGripperModel("hands", "dexs_gripper")

    @property
    def f5d6_hand(self) -> F5d6HandModel:
        return F5d6HandModel("hands", "f5d6_hand")


humanoid = HumanoidType("humanoid")
hands = HandsType("hands")


def get_all_robot_dirs() -> list[RobotModel]:
    return []
