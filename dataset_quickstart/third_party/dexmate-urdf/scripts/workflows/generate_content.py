# Copyright 2025 Dexmate Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to generate Python implementation files for better auto-completion."""

import ast
from pathlib import Path
from typing import Any


def to_pascal_case(name: str) -> str:
    """Convert a string to PascalCase."""
    # Replace hyphens and underscores with spaces, then capitalize each word
    return "".join(word.capitalize() for word in name.replace("-", "_").split("_"))


def to_python_identifier(name: str) -> str:
    """Convert a string to a valid Python identifier."""
    # Replace hyphens and dots with underscores
    return name.replace("-", "_").replace(".", "_")


def get_base_model_name(model_name: str) -> str:
    """Extract the base model name from a full model name."""
    # For example, from 'vega-rc1-right-arm' return 'vega'
    return model_name.split("-")[0]


def create_import(name: str, asname: str | None = None) -> ast.Import:
    """Create an import statement."""
    return ast.Import(names=[ast.alias(name=name, asname=asname)])


def create_import_from(module: str, names: list[str]) -> ast.ImportFrom:
    """Create a from-import statement."""
    return ast.ImportFrom(
        module=module,
        names=[ast.alias(name=name, asname=None) for name in names],
        level=0,
    )


def create_class_def(name: str, bases: list[str], body: list[ast.stmt]) -> ast.ClassDef:
    """Create a class definition."""
    # Use direct class references instead of string literals
    base_nodes = [ast.Name(id=base, ctx=ast.Load()) for base in bases]
    return ast.ClassDef(
        name=name, bases=base_nodes, keywords=[], body=body, decorator_list=[]
    )


def create_method_def(
    name: str, args: list[str], body: list[ast.stmt], returns: ast.expr | None = None
) -> ast.FunctionDef:
    """Create a method definition."""
    return ast.FunctionDef(
        name=name,
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=arg, annotation=None) for arg in args],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=body,
        decorator_list=[],
        returns=returns,
    )


def create_property(
    name: str, return_type: str, implementation: list[ast.stmt]
) -> ast.FunctionDef:
    """Create a property definition with actual implementation."""
    return ast.FunctionDef(
        name=name,
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="self", annotation=None)],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=implementation,
        decorator_list=[ast.Name(id="property", ctx=ast.Load())],
        returns=ast.Name(id=return_type, ctx=ast.Load()),
    )


def create_assign(target: str, value: Any) -> ast.Assign:
    """Create an assignment statement."""
    return ast.Assign(targets=[ast.Name(id=target, ctx=ast.Store())], value=value)


def create_string_literal(value: str) -> ast.Constant:
    """Create a string literal."""
    return ast.Constant(value=value)


def create_list_literal(elements: list[ast.expr]) -> ast.List:
    """Create a list literal."""
    return ast.List(elts=elements, ctx=ast.Load())


def generate_content():
    """Generate Python implementation files for the robot module."""
    robot_dir = Path(__file__).parent.parent.parent / "src/dexmate_urdf/robots"
    if not robot_dir.exists():
        print(f"Robot directory not found: {robot_dir}")
        return

    # Collect information about the robot structure
    robot_types: dict[str, list[str]] = {}
    robot_models: dict[str, dict[str, list[str]]] = {}

    # Scan the directory structure (simplified - no version directories)
    for type_dir in robot_dir.iterdir():
        if type_dir.is_dir() and type_dir.name != "__pycache__":
            type_name = type_dir.name
            robot_types[type_name] = []
            robot_models[type_name] = {}

            for model_dir in type_dir.iterdir():
                if model_dir.is_dir():
                    model_name = model_dir.name
                    robot_types[type_name].append(model_name)
                    robot_models[type_name][model_name] = []

                    # Collect URDF files directly from model directory
                    for file in model_dir.iterdir():
                        if (
                            file.suffix == ".urdf"
                            and not file.name.endswith(".collision.urdf")
                            and ".collision_" not in file.name
                        ):
                            robot_models[type_name][model_name].append(file.name)

    # Generate AST nodes for only the specific classes
    module_body = [
        # Add necessary imports
        create_import_from(".paths", ["RobotModel", "RobotType", "URDFModel"]),
    ]

    # Add specific RobotModel classes (simplified structure)
    for type_name, models in robot_models.items():
        for model_name, urdf_files in models.items():
            class_name = to_pascal_case(model_name) + "Model"
            model_class_body = []

            # Add URDF model properties directly to the model class
            for urdf_file in urdf_files:
                attr_name = to_python_identifier(urdf_file.rsplit(".", 1)[0])
                model_class_body.append(
                    create_property(
                        attr_name,
                        "URDFModel",
                        [
                            ast.Return(
                                value=ast.Call(
                                    func=ast.Name(id="URDFModel", ctx=ast.Load()),
                                    args=[
                                        ast.Attribute(
                                            value=ast.Name(id="self", ctx=ast.Load()),
                                            attr="_type",
                                            ctx=ast.Load(),
                                        ),
                                        ast.Attribute(
                                            value=ast.Name(id="self", ctx=ast.Load()),
                                            attr="_name",
                                            ctx=ast.Load(),
                                        ),
                                        create_string_literal(attr_name),
                                    ],
                                    keywords=[],
                                )
                            )
                        ],
                    )
                )

            # Ensure non-empty body
            if not model_class_body:
                model_class_body.append(ast.Pass())

            module_body.append(
                create_class_def(class_name, ["RobotModel"], model_class_body)
            )

    # Add specific RobotType classes
    for type_name, models in robot_models.items():
        type_class_name = f"{to_pascal_case(type_name)}Type"
        type_class_body = []
        for model_name in models:
            model_class_name = f"{to_pascal_case(model_name)}Model"
            type_class_body.append(
                create_property(
                    model_name,
                    model_class_name,
                    [
                        ast.Return(
                            value=ast.Call(
                                func=ast.Name(id=model_class_name, ctx=ast.Load()),
                                args=[
                                    create_string_literal(type_name),
                                    create_string_literal(model_name),
                                ],
                                keywords=[],
                            )
                        )
                    ],
                )
            )
        # Ensure non-empty body
        if not type_class_body:
            type_class_body.append(ast.Pass())
        # Use direct inheritance for RobotType
        module_body.append(
            create_class_def(type_class_name, ["RobotType"], type_class_body)
        )

    # Add global variables
    for type_name in robot_types:
        type_class_name = f"{to_pascal_case(type_name)}Type"
        module_body.append(
            create_assign(
                type_name,
                ast.Call(
                    func=ast.Name(id=type_class_name, ctx=ast.Load()),
                    args=[create_string_literal(type_name)],
                    keywords=[],
                ),
            )
        )

    # Add get_all_robot_dirs function
    module_body.append(
        create_method_def(
            "get_all_robot_dirs",
            [],
            [ast.Return(value=ast.List(elts=[], ctx=ast.Load()))],
            returns=ast.Subscript(
                value=ast.Name(id="list", ctx=ast.Load()),
                slice=ast.Name(id="RobotModel", ctx=ast.Load()),
                ctx=ast.Load(),
            ),
        )
    )

    # Create the module and generate code
    content_path = robot_dir / "content.py"
    if module_body:
        module = ast.Module(body=module_body, type_ignores=[])
        ast.fix_missing_locations(module)
        code = ast.unparse(module)

        # Write the generated code to file
        with open(content_path, "w") as f:
            f.write(code)

        # Format the file using ruff
        import subprocess

        subprocess.run(["ruff", "format", str(content_path)], check=True)

        print(f"Generated Python implementation file: {content_path}")
    else:
        print("No robot models found to generate.")


if __name__ == "__main__":
    generate_content()
