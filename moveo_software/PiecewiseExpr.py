import sympy as sp
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union


@dataclass
class Piece:
    expression: sp.Expr
    lower_bound: Union[sp.core.numbers.NegativeInfinity, float]
    upper_bound: Union[sp.core.numbers.Infinity, float]


class PiecewiseFunction:
    def __init__(
        self, pieces: List[Piece], indep_var_sym: str = "x", dep_var_sym: str = "y"
    ):
        self.pieces = pieces
        self.id_var = sp.Symbol(indep_var_sym)
        self.d_var = sp.Symbol(dep_var_sym)
        self.expr = self._build_piecewise_function()

    def _build_piecewise_function(self):
        try:
            piecewise_expr = []
            for piece in self.pieces:
                expr, lower, upper = (
                    piece.expression,
                    piece.lower_bound,
                    piece.upper_bound,
                )
                if lower == sp.core.numbers.NegativeInfinity:
                    cond = self.id_var < upper
                elif upper == sp.core.numbers.Infinity:
                    cond = self.id_var >= lower
                else:
                    cond = (self.id_var >= lower) & (self.id_var < upper)
                piecewise_expr.append((expr, cond))
            return sp.Piecewise(*piecewise_expr)

        except TypeError as e:
            return sp.Piecewise((0, True))

    def integrate(self, first_integration_ctt=0):

        integrated_pieces = []
        integration_constants = [first_integration_ctt]

        for i in range(1, len(self.pieces)):
            prev_piece = self.pieces[i - 1]
            curr_piece = self.pieces[i]
            prev_integration_ctt = integration_constants[i - 1]

            prev_piece_int = sp.integrate(prev_piece.expression, self.id_var)
            curr_piece_int = sp.integrate(curr_piece.expression, self.id_var)

            prev_piece_upp_val = prev_piece_int.subs(
                self.id_var, prev_piece.upper_bound
            )
            curr_piece_low_val = curr_piece_int.subs(
                self.id_var, curr_piece.lower_bound
            )

            current_integration_ctt = (
                prev_piece_upp_val + prev_integration_ctt - curr_piece_low_val
            )
            integration_constants.append(current_integration_ctt)

            actual_curr_piece_int_expr = curr_piece_int + current_integration_ctt

            integrated_pieces.append(
                Piece(
                    actual_curr_piece_int_expr,
                    curr_piece.lower_bound,
                    curr_piece.upper_bound,
                )
            )

        return PiecewiseFunction(integrated_pieces, str(self.id_var), str(self.d_var))

    def derive(self):
        derived_pieces = []
        for piece in self.pieces:
            derived_expr = sp.diff(piece.expression, self.id_var)
            derived_pieces.append(
                Piece(
                    derived_expr,
                    piece.lower_bound,
                    piece.upper_bound,
                )
            )
        return PiecewiseFunction(derived_pieces, str(self.id_var), str(self.d_var))

    def subs_IndepVar(self, x_value):
        """
        Evaluate the piecewise function at a specific value of the independent variable.
        """
        x_value = float(x_value)

        for piece in self.pieces:

            lower, upper = piece.lower_bound, piece.upper_bound

            if (lower == sp.core.numbers.NegativeInfinity or x_value >= lower) and (
                upper == sp.core.numbers.Infinity or x_value < upper
            ):

                return piece.expression.subs(self.id_var, x_value)

        raise ValueError(
            "The provided value is outside the bounds of the piecewise function."
        )

    def subs_DependVar(self, dep_var_value):
        solutions = []
        for piece in self.pieces:
            # Skip the piece if the expression is zero and dep_var_value is not zero
            if piece.expression == 0 and dep_var_value != 0:
                continue

            # Solve the equation only if piece.expression is not zero
            if piece.expression != 0:
                # Solve the equation piece.expression = dep_var_value for the independent variable
                sol = sp.solve(piece.expression - dep_var_value, self.id_var)

                # Check each solution to ensure it is real and within the bounds of the piece
                for s in sol:
                    if s.is_real and piece.lower_bound < s <= piece.upper_bound:
                        solutions.append(s)

        # Return the first valid solution if there are any
        if solutions:
            return solutions[0]
        # Special case handling when no solutions are found and dep_var_value is zero
        elif not solutions and dep_var_value == 0:
            return 0
        else:
            return None

    def plot(
        self,
        t_min,
        t_max,
        num_points=500,
        title=None,
        label=None,
        show=False,
        wrap=False,
        wrap_limits=[0, 360],
        xlabel="xlabel",
        ylabel="ylabel",
        color=None,
    ):
        plt.style.use("default")

        v_func = sp.lambdify(self.id_var, self.expr, "numpy")
        t_vals = np.linspace(t_min, t_max, num_points)
        v_vals = v_func(t_vals)

        if wrap:
            lower_limit, upper_limit = wrap_limits
            range_span = upper_limit - lower_limit

            # Wrap the values within the specified range
            v_vals = (v_vals - lower_limit) % range_span + lower_limit

            # Normalize the values to the range [0, 1]
            v_vals = (v_vals - lower_limit) / range_span

            plt.ylim(bottom=0, top=1)

        else:
            pass

        # Plot with color if specified
        if color:
            plt.plot(t_vals, v_vals, label=label, color=color)
        else:
            plt.plot(t_vals, v_vals, label=label)

        # Add axis labels
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if show:
            plt.axhline(0, color="black", linewidth=0.5)
            plt.axvline(0, color="black", linewidth=0.5)
            plt.grid(color="gray", linestyle="--", linewidth=0.5)
            plt.title(title)
            plt.legend()
            plt.show()


if __name__ == "__main__":

    np.set_printoptions(precision=2, suppress=True)

    v_max = 10  # Maximum velocity
    t_acc = 2  # Time when acceleration ends and constant velocity begins
    t_dec = 4  # Time when deceleration starts
    t_end = 6  # Time when motion ends

    vel_pieces = [
        Piece(0, sp.core.numbers.NegativeInfinity, 0),
        Piece(v_max * sp.Symbol("t") / t_acc, 0, t_acc),
        Piece(v_max, t_acc, t_dec),
        Piece(v_max * (1 - (sp.Symbol("t") - t_dec) / (t_end - t_dec)), t_dec, t_end),
        Piece(0, t_end, sp.core.numbers.Infinity),
    ]

    pwf = PiecewiseFunction(vel_pieces, "t")

    # Plot the original piecewise function
    plt.figure()
    pwf.plot(-1, t_end + 1, title="Velocity", show=True)

    # Integrate and plot the resulting expression
    integrated_pwf = pwf.integrate()
    plt.figure()
    integrated_pwf.plot(-1, t_end + 1, title="Position", show=True)

    # Derive and plot the resulting expression
    derived_pwf = pwf.derive()
    plt.figure()
    derived_pwf.plot(-1, t_end + 1, title="Acceleration", show=True)
