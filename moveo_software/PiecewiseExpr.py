import sympy as sp
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union


@dataclass
class Piece:
    """
    Represents a single piece of a piecewise function, consisting of a mathematical
    expression and bounds for the independent variable.

    Attributes:
    - expression (sp.Expr): The mathematical expression defining the piece of the function.
    - lower_bound (Union[sp.core.numbers.NegativeInfinity, float]): The lower bound
        of the independent variable for this piece. It can be negative infinity.
    - upper_bound (Union[sp.core.numbers.Infinity, float]): The upper bound of the
        independent variable for this piece. It can be positive infinity.
    """

    expression: sp.Expr
    lower_bound: Union[sp.core.numbers.NegativeInfinity, float]
    upper_bound: Union[sp.core.numbers.Infinity, float]


class PiecewiseFunction:
    """
    A class to represent a piecewise function, composed of different mathematical
    expressions for different ranges of an independent variable.

    Attributes:
    - pieces (List[Piece]): A list of pieces where each piece contains a mathematical
                            expression and a condition for the independent variable.
    - id_var (sympy.Symbol): The independent variable symbol (default is "x").
    - d_var (sympy.Symbol): The dependent variable symbol (default is "y").
    - expr (sympy.Piecewise): The final piecewise expression constructed from the
                            list of pieces.

    """

    def __init__(
        self, pieces: List[Piece], indep_var_sym: str = "x", dep_var_sym: str = "y"
    ):
        """
        Initializes the PiecewiseFunction with the provided pieces and variable symbols.

        Parameters:
        - pieces (List[Piece]): A list of pieces where each piece contains a mathematical
            expression and a condition for the independent variable.
        - indep_var_sym (str): The independent variable symbol (default is "x").
        - dep_var_sym (str): The dependent variable symbol (default is "y").
        """
        self.pieces = pieces  # Store the list of pieces.
        self.id_var = sp.Symbol(
            indep_var_sym
        )  # Create a symbolic independent variable.
        self.d_var = sp.Symbol(dep_var_sym)  # Create a symbolic dependent variable.
        self.expr = (
            self._build_piecewise_function()
        )  # Build the full piecewise function.

    def _build_piecewise_function(self):
        """
        Builds a SymPy Piecewise function from the provided pieces.

        Each piece in the list is a mathematical expression with associated bounds
        (lower and upper) for the independent variable. The method constructs
        conditions for each piece, determining where the expression is valid.

        Returns:
        - sp.Piecewise: A SymPy Piecewise function combining all the pieces
            and their conditions.

        Raises:
        - TypeError: If the pieces are not structured correctly, a default
            constant piecewise function (0 for all conditions) is returned.
        """
        try:
            piecewise_expr = (
                []
            )  # Initialize an empty list to store the expression and condition pairs.
            for (
                piece
            ) in self.pieces:  # Loop through each piece to extract its components.
                expr, lower, upper = (
                    piece.expression,  # The mathematical expression for the piece.
                    piece.lower_bound,  # The lower bound of the independent variable for this piece.
                    piece.upper_bound,  # The upper bound of the independent variable for this piece.
                )
                # Define the condition based on the bounds
                if lower == sp.core.numbers.NegativeInfinity:
                    cond = self.id_var < upper  # No lower bound, just upper limit.
                elif upper == sp.core.numbers.Infinity:
                    cond = self.id_var >= lower  # No upper bound, just a lower limit.
                else:
                    cond = (self.id_var >= lower) & (
                        self.id_var < upper
                    )  # Both bounds are finite.

                # Append the expression and its condition as a tuple.
                piecewise_expr.append((expr, cond))

            # Return the constructed SymPy Piecewise function.
            return sp.Piecewise(*piecewise_expr)

        except TypeError as e:
            # If there's a TypeError (e.g., wrong structure in the pieces), return a default function.
            return sp.Piecewise((0, True))  # Default: 0 for all conditions (catch-all).

    def integrate(self, first_integration_ctt=0):
        """
        Integrates the piecewise function with respect to the independent variable.

        The integration is performed piecewise, ensuring that the integration constants
        are adjusted to make the resulting function continuous across boundaries between pieces.

        Parameters:
        - first_integration_ctt (float, optional): The integration constant to use for
            the first piece of the function. Defaults to 0.

        Returns:
        - PiecewiseFunction: A new PiecewiseFunction object where each piece represents
            the integral of the corresponding piece in the original function.
        """

        integrated_pieces = []  # List to hold the integrated pieces.
        integration_constants = [
            first_integration_ctt
        ]  # Track integration constants across pieces.

        # Loop through each piece starting from the second piece to adjust integration constants.
        for i in range(1, len(self.pieces)):
            prev_piece = self.pieces[i - 1]  # Previous piece in the list.
            curr_piece = self.pieces[i]  # Current piece being integrated.
            prev_integration_ctt = integration_constants[
                i - 1
            ]  # Previous integration constant.

            # Integrate the expressions of the previous and current pieces.
            prev_piece_int = sp.integrate(prev_piece.expression, self.id_var)
            curr_piece_int = sp.integrate(curr_piece.expression, self.id_var)

            # Evaluate the upper bound of the previous piece after integration.
            prev_piece_upp_val = prev_piece_int.subs(
                self.id_var, prev_piece.upper_bound
            )
            # Evaluate the lower bound of the current piece after integration.
            curr_piece_low_val = curr_piece_int.subs(
                self.id_var, curr_piece.lower_bound
            )

            # Adjust the integration constant to ensure continuity between pieces.
            current_integration_ctt = (
                prev_piece_upp_val + prev_integration_ctt - curr_piece_low_val
            )
            integration_constants.append(current_integration_ctt)

            # Add the adjusted integration constant to the current integrated piece.
            actual_curr_piece_int_expr = curr_piece_int + current_integration_ctt

            # Create a new piece with the integrated expression and add it to the list.
            integrated_pieces.append(
                Piece(
                    actual_curr_piece_int_expr,
                    curr_piece.lower_bound,
                    curr_piece.upper_bound,
                )
            )

        # Return a new PiecewiseFunction with the integrated pieces and same variable symbols.
        return PiecewiseFunction(integrated_pieces, str(self.id_var), str(self.d_var))

    def derive(self):
        """
        Differentiates the piecewise function with respect to the independent variable.

        The differentiation is performed piecewise for each segment of the function, with
        each segment's derivative being calculated independently.

        Returns:
        - PiecewiseFunction: A new PiecewiseFunction object where each piece represents
            the derivative of the corresponding piece in the original function.
        """

        derived_pieces = []  # List to hold the differentiated pieces.

        # Loop through each piece and differentiate its expression.
        for piece in self.pieces:
            derived_expr = sp.diff(
                piece.expression, self.id_var
            )  # Differentiate the piece.

            # Create a new piece with the differentiated expression and the same bounds.
            derived_pieces.append(
                Piece(
                    derived_expr,
                    piece.lower_bound,
                    piece.upper_bound,
                )
            )

        # Return a new PiecewiseFunction with the differentiated pieces and same variable symbols.
        return PiecewiseFunction(derived_pieces, str(self.id_var), str(self.d_var))

    def subs_IndepVar(self, x_value):
        """
        Evaluate the piecewise function at a specific value of the independent variable.

        This method substitutes a given value for the independent variable and returns
        the corresponding value of the function based on the piecewise segments.

        Parameters:
        - x_value (float): The value at which the piecewise function is to be evaluated.

        Returns:
        - float: The evaluated value of the function at the given x_value.

        Raises:
        - ValueError: If the x_value is outside the bounds of the piecewise function.
        """

        x_value = float(x_value)  # Ensure the input value is a float.

        # Loop through each piece to find the appropriate range for x_value.
        for piece in self.pieces:
            lower, upper = piece.lower_bound, piece.upper_bound

            # Check if the x_value falls within the bounds of the current piece.
            if (lower == sp.core.numbers.NegativeInfinity or x_value >= lower) and (
                upper == sp.core.numbers.Infinity or x_value < upper
            ):
                # If so, substitute the x_value into the expression and return the result.
                return piece.expression.subs(self.id_var, x_value)

        # If no appropriate piece was found, raise an error indicating out-of-bounds input.
        raise ValueError(
            "The provided value is outside the bounds of the piecewise function."
        )

    def subs_DependVar(self, dep_var_value):
        """
        Solve for the independent variable given a specific value of the dependent variable.

        This method finds the value(s) of the independent variable where the piecewise
        function equals the specified dependent variable value. It checks each piece for
        possible solutions and returns the first valid one.

        Parameters:
        - dep_var_value (float): The value of the dependent variable to solve for.

        Returns:
        - float: The first valid solution for the independent variable where the function
            equals dep_var_value.
        - None: If no valid solution is found and dep_var_value is not zero.
        - 0: If dep_var_value is zero and no other solutions are found (special case).
        """

        solutions = []  # List to store valid solutions for the independent variable.

        # Loop through each piece to solve the equation piece.expression = dep_var_value.
        for piece in self.pieces:
            # Skip the piece if the expression is zero and the dep_var_value is not zero.
            if piece.expression == 0 and dep_var_value != 0:
                continue

            # Only attempt to solve if the piece's expression is non-zero.
            if piece.expression != 0:
                # Solve the equation piece.expression = dep_var_value for the independent variable.
                sol = sp.solve(piece.expression - dep_var_value, self.id_var)

                # Check each solution to ensure it is real and within the piece's bounds.
                for s in sol:
                    if s.is_real and piece.lower_bound < s <= piece.upper_bound:
                        solutions.append(s)  # Add valid solutions to the list.

        # Return the first valid solution if any exist.
        if solutions:
            return solutions[0]
        # Special case: if no solutions are found and dep_var_value is zero, return 0.
        elif not solutions and dep_var_value == 0:
            return 0
        # Return None if no valid solutions are found and dep_var_value is not zero.
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
        """
        Plot the piecewise function over a specified range of the independent variable.

        The method evaluates the piecewise function over a range of values for the
        independent variable and plots the resulting function. It also supports optional
        wrapping of the function values within a specified range (useful for periodic functions).

        Parameters:
        - t_min (float): The minimum value of the independent variable for the plot.
        - t_max (float): The maximum value of the independent variable for the plot.
        - num_points (int, optional): The number of points to evaluate between t_min and t_max.
            Defaults to 500.
        - title (str, optional): The title of the plot. Defaults to None.
        - label (str, optional): The label for the function to be displayed in the legend.
            Defaults to None.
        - show (bool, optional): Whether to display the plot immediately. Defaults to False.
        - wrap (bool, optional): Whether to wrap the function values within a specified range.
            Useful for periodic functions. Defaults to False.
        - wrap_limits (list, optional): The lower and upper limits to wrap the function values.
            Only used if wrap is True. Defaults to [0, 360].
        - xlabel (str, optional): Label for the x-axis. Defaults to "xlabel".
        - ylabel (str, optional): Label for the y-axis. Defaults to "ylabel".
        - color (str, optional): The color to use for the plot line. Defaults to None.

        Returns:
        - None
        """

        # Set the plot style to the default.
        plt.style.use("default")

        # Convert the piecewise function to a numpy-compatible function for plotting.
        v_func = sp.lambdify(self.id_var, self.expr, "numpy")

        # Generate evenly spaced values of the independent variable.
        t_vals = np.linspace(t_min, t_max, num_points)

        # Evaluate the piecewise function at each of the generated values.
        v_vals = v_func(t_vals)

        # If wrapping is enabled, adjust the values to fit within the specified range.
        if wrap:
            lower_limit, upper_limit = wrap_limits
            range_span = upper_limit - lower_limit

            # Wrap the values within the specified range.
            v_vals = (v_vals - lower_limit) % range_span + lower_limit

            # Normalize the wrapped values to the range [0, 1].
            v_vals = (v_vals - lower_limit) / range_span

            # Set y-axis limits for the wrapped plot.
            plt.ylim(bottom=0, top=1)

        # Plot the function values.
        if color:
            plt.plot(
                t_vals, v_vals, label=label, color=color
            )  # Plot with specified color.
        else:
            plt.plot(t_vals, v_vals, label=label)  # Plot with default color.

        # Add axis labels to the plot.
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # If 'show' is True, add extra plot features and display the plot.
        if show:
            plt.axhline(
                0, color="black", linewidth=0.5
            )  # Add a horizontal line at y=0.
            plt.axvline(0, color="black", linewidth=0.5)  # Add a vertical line at x=0.
            plt.grid(
                color="gray", linestyle="--", linewidth=0.5
            )  # Add a grid to the plot.
            plt.title(title)  # Add the plot title if provided.
            plt.legend()  # Display the legend if labels are provided.
            plt.show()  # Show the plot window.


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
