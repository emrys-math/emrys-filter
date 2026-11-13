#pragma once

/** \file
 * Printing: \c operator<< and \c std::formatter for every distribution, every filter (which
 * prints its name and nests its state under it), and the library's error codes, so that
 * <tt>std::cout << filter</tt> and <tt>std::println("{}", filter.state())</tt> work. Each type
 * describes itself through an internal::describe() overload beside it, and this header turns any
 * type with one into an internal::Printable, which the operator and the formatter take.
 *
 * A description is a summary, never the elements: a particle set prints its count, its effective
 * sample size, and its weighted mean and covariance, not its particles, which its \c states()
 * and \c logWeights() print through Eigen. Its first line names the distribution and its
 * dimension, and the lines after it are indented by two spaces per level, so a mixture's
 * components nest under it. Numbers follow the stream's precision, as Eigen's do; the formatter
 * takes a width, a fill, and an alignment, as a string does, and a precision, <tt>{:.3}</tt>, as
 * the stream's.
 *
 * Defining \c EMRYS_FILTER_NO_IO before the first EmrysFilter header leaves all of it out, with
 * the standard headers \c format, \c ostream, and \c sstream; Eigen's \c EIGEN_NO_IO implies it (Macros.h).
 *
 * Read \ref Spaces next.
 */

#include <EmrysFilter/Core/Macros.h>

#ifndef EMRYS_FILTER_NO_IO

#    include <expected>
#    include <format>
#    include <ostream>
#    include <sstream>
#    include <string>
#    include <string_view>

#    include <EmrysFilter/Core/Weights.h>

namespace Emrys
{

namespace internal
{

/** How deeply a line of a description is indented: two spaces per level. Every describe()
 * overload takes one, which is also what makes `internal` a namespace that argument-dependent
 * lookup searches for them.
 */
struct Indent
{
    int depth = 0;  ///< the level, from 0

    /** \returns the level below this one */
    [[nodiscard]] Indent
    deeper() const
    {
        return {depth + 1};
    }
};

/** Writes the spaces of \a indent. */
inline std::ostream &
operator<<(std::ostream & stream, Indent indent)
{
    for (int i = 0; i < indent.depth; ++i)
    {
        stream << "  ";
    }
    return stream;
}

/** Whether the library prints \a T: whether an internal::describe() overload takes it, found by
 * argument-dependent lookup through Indent. A type opts in by having one, so neither Eigen's
 * matrices nor a user's types are claimed by the library's \c operator<< or \c std::formatter.
 */
template <typename T>
concept Printable = requires(std::ostream & stream, T const & value) { describe(stream, value, Indent{}); };

/** Writes \a label and the coefficients of \a vector on one line at \a indent, separated by
 * spaces.
 */
template <typename Derived>
void
printRow(std::ostream & stream, Indent indent, std::string_view label, Eigen::DenseBase<Derived> const & vector)
{
    static auto const row = Eigen::IOFormat{Eigen::StreamPrecision, Eigen::DontAlignCols, " ", " "};
    stream << indent << label << ':';
    if (vector.size() > 0)
    {
        stream << ' ' << vector.reshaped().transpose().format(row);
    }
}

/** Writes \a label on a line at \a indent and the rows of \a matrix under it, one level deeper,
 * with their columns aligned.
 */
template <typename Derived>
void
printMatrix(std::ostream & stream, Indent indent, std::string_view label, Eigen::DenseBase<Derived> const & matrix)
{
    stream << indent << label << ':';
    if (matrix.size() > 0)
    {
        auto prefix = std::ostringstream{};
        prefix << indent.deeper();
        auto const rows = Eigen::IOFormat{Eigen::StreamPrecision, 0, " ", "\n", prefix.str()};
        stream << '\n' << matrix.format(rows);
    }
}

/** Writes " of dimension n", and the difference dimension after it where it differs. */
inline void
printDimension(std::ostream & stream, Eigen::Index valueDimension, Eigen::Index differenceDimension)
{
    stream << " of dimension " << valueDimension;
    if (differenceDimension != valueDimension)
    {
        stream << " (" << differenceDimension << " in differences)";
    }
}

/** Writes \a count and \a noun, adding "s" unless \a count is one. */
inline void
printCount(std::ostream & stream, Eigen::Index count, std::string_view noun)
{
    stream << count << ' ' << noun << (count == 1 ? "" : "s");
}

/** Writes the mean and covariance of \a moments, a Moments, under the line before them at
 * \a indent: a line of the mean and the covariance's rows under a line of its own.
 */
template <typename M>
void
printMoments(std::ostream & stream, Indent indent, M const & moments)
{
    stream << '\n';
    printRow(stream, indent, "mean", moments.mean());
    stream << '\n';
    printMatrix(stream, indent, "covariance", moments.covarianceLower());
}

/** printMoments() of the moments in \a moments, or a line naming the error where there are
 * none.
 */
template <typename M, typename E>
void
printMoments(std::ostream & stream, Indent indent, std::expected<M, E> const & moments)
{
    if (moments)
    {
        printMoments(stream, indent, *moments);
    }
    else
    {
        stream << '\n' << indent << "no moments: ";
        describe(stream, moments.error(), Indent{});
    }
}

/** Describes a filter: \a name on a line at \a indent, and the description of \a state, the
 * filter's, nested under it, as a mixture nests its components.
 */
template <typename State>
void
describeFilter(std::ostream & stream, std::string_view name, State const & state, Indent indent)
{
    stream << indent << name << '\n';
    describe(stream, state, indent.deeper());
}

/** Describes a NumericError, a GaussianError or a WeightError, in words. */
inline void
describe(std::ostream & stream, NumericError error, Indent indent)
{
    stream << indent;
    switch (error)
    {
        case NumericError::NotPositiveDefinite:
            stream << "the covariance is not positive definite";
            break;
        case NumericError::Singular:
            stream << "the precision is singular";
            break;
        case NumericError::NonFinite:
            stream << "a value is not finite";
            break;
        case NumericError::Empty:
            stream << "no weights";
            break;
        case NumericError::ZeroSum:
            stream << "every weight is zero";
            break;
    }
}

}  // namespace internal

/** \ingroup Core
 * Prints \a value, any type the library describes (internal::Printable): a distribution, a
 * filter, by name with its state nested under it, or an error code. See Print.h for what a description
 * holds.
 *
 * \code
 * std::cout << filter << '\n';
 * \endcode
 */
template <internal::Printable T>
std::ostream &
operator<<(std::ostream & stream, T const & value)
{
    describe(stream, value, internal::Indent{});
    return stream;
}

}  // namespace Emrys

/** \ingroup Core
 * \brief Formats any type the library describes (Emrys::internal::Printable) as its
 * \c operator<< prints it.
 *
 * The specification is a string's, but for its precision: a width, a fill, and an alignment
 * apply to the whole description, and a precision is the stream's, the significant digits of its
 * numbers, rather than the string's truncation. Without one the numbers are at the stream's
 * default precision. The precision is a number, not a replacement field, and ends the
 * specification but for a type \c s; a width given by a replacement field takes no precision.
 *
 * \code
 * std::println("{}", filter.state());
 * std::println("{:.3}", filter.state());  // three significant digits
 * \endcode
 */
template <Emrys::internal::Printable T>
struct std::formatter<T, char> : std::formatter<std::string_view, char>
{
    /** Parses \a context's specification: a precision at its end is the stream's, and the rest is
     * the string's.
     */
    template <typename ParseContext>
    constexpr typename ParseContext::iterator
    parse(ParseContext & context)
    {
        auto const begin = context.begin();
        auto end = begin;
        while (end != context.end() && *end != '}')
        {
            ++end;
        }
        auto const spec = std::string_view{begin, end};
        // The spec without a trailing type s, which is a string's default.
        auto const body = spec.ends_with('s') ? spec.substr(0, spec.size() - 1) : spec;
        // A precision is a '.' and the digits after it, at the end. A '.' as the fill is followed
        // by an alignment, so a '.' just before the last digits is always the precision.
        auto digits = body.size();
        while (digits > 0 && body[digits - 1] >= '0' && body[digits - 1] <= '9')
        {
            --digits;
        }
        if (digits == body.size() || digits == 0 || body[digits - 1] != '.')
        {
            return std::formatter<std::string_view, char>::parse(context);
        }
        int precision = 0;
        for (char const digit : body.substr(digits))
        {
            precision = precision * 10 + (digit - '0');
        }
        m_precision = precision;
        // The string's part, the spec before the precision, parsed on its own.
        auto rest = std::format_parse_context{body.substr(0, digits - 1)};
        std::formatter<std::string_view, char>::parse(rest);
        return end;
    }

    /** Writes the description of \a value to \a context's output, its numbers at the precision
     * given, if one was.
     */
    template <typename Context>
    [[nodiscard]] typename Context::iterator
    format(T const & value, Context & context) const
    {
        auto stream = std::ostringstream{};
        if (m_precision >= 0)
        {
            stream.precision(m_precision);
        }
        stream << value;
        return std::formatter<std::string_view, char>::format(stream.str(), context);
    }

private:

    int m_precision = -1;  ///< the stream's precision, or -1 for its default
};

#endif  // EMRYS_FILTER_NO_IO
